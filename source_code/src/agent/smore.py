import copy
import numpy as np
import time
import torch

from mpi4py import MPI
from src.gcrl_model import *
from src.replay_buffer import ReplayBuffer
from src.goal_utils import *
from src.agent.base import Agent

def iql_loss(pred, target, expectile=0.5):
    err = target - pred
    weight = torch.abs(expectile - (err < 0).float())
    return weight * torch.square(err)

"""
SMORE

"""
class SMORE(Agent):
    def __init__(self, args, env):
        super().__init__(args, env) 
        # create the network
        self.args = args
        self.args.gamma = 0.99
        self.temperature = 3.0
        self.clip_score = 10
        self.expectile = 0.7
        self.args.gcqs = True
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor
        self.actor = GaussianActor(self.args).to(self.device)
        sync_networks(self.actor)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_actor)
        self.actor_target = GaussianActor(self.args).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        #Crtic
        self.critic = EnsembleContinuousMLPCritic(self.args).to(self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_critic)
        self.critic_target = EnsembleContinuousMLPCritic(self.args).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Value
        self.value 		= MLPValue(self.args).to(self.device)
        sync_networks(self.value)
        self.value_target 	= MLPValue(self.args).to(self.device)
        self.value_target.load_state_dict(self.value.state_dict())
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.args.lr_critic)

        self.lock = threading.Lock()

        self.sample_gcqs_transitions =  self.sampler.sample_gcqs_transitions

        def sample_func(S, A, AG,  G, size):
            return self.sample_gcqs_transitions(
                    S, A, AG, G, size)
        self.sample_func = sample_func

        # Configure the replay buffer.
        self.buffer = ReplayBuffer(self.args, self.sample_func)

    # update the network
    def _update(self):
        transition = self.sample_batch()
        S  = transition['S']
        NS = transition['NS']
        A  = transition['A']
        G  = transition['G']
        R  = transition['R']

        R = numpy2torch(R, unsqueeze=False, cuda=self.args.cuda)
        A = numpy2torch(A, unsqueeze=False, cuda=self.args.cuda)
        S, G = self._preproc_inputs(S, G)
        NS, _ = self._preproc_inputs(NS)

        # First compute the value loss
        with torch.no_grad():
            target_q = self.critic_target(S, A, G)
            target_q = torch.min(target_q, dim=0)[0]
        vs = self.value(S, G)  # Always detach for value learning
        v_loss = iql_loss(vs, target_q.expand(vs.shape[0], -1), self.expectile).mean(dim=-1).sum()
         
        neg_r = -2
        # Next, compute the critic loss
        with torch.no_grad():
            next_vs = self.value(NS, G)
            next_v = torch.min(next_vs, dim=0)[0]
            target = neg_r + self.args.gamma * next_v

        qs = self.critic(
            S, A, G
        )
        
        def stable_loss(q, q_target):
            alpha_coeff = (1-self.args.gamma)*1e-4
            loss = ((q[q.shape[0]//2:]+neg_r/(1-self.args.gamma))**2).mean()+alpha_coeff*((q[:q.shape[0]//2]-0)**2).mean()+0.5*((q[:q.shape[0]//2]-q_target[:q.shape[0]//2])**2).mean()

            return loss.mean()
        
        q_loss = stable_loss(qs, target.expand(qs.shape[0], -1))

        # Next, update the actor. We detach and use the old value, v for computational efficiency
        # though the JAX IQL recomputes it, while Pytorch IQL versions do not.
        with torch.no_grad():
            adv = target_q - torch.min(vs, dim=0)[0]
            exp_adv = torch.exp(adv * self.temperature)
            if self.clip_score is not None:
                exp_adv = torch.clamp(exp_adv, max=self.clip_score)

        dist = self.actor(S, G)
        if isinstance(dist, torch.distributions.Distribution):
            bc_loss = -dist.log_prob(A).sum(dim=-1)
        elif torch.is_tensor(dist):
            assert dist.shape == actions.shape
            bc_loss = torch.nn.functional.mse_loss(dist, A, reduction="none").sum(dim=-1)
        else:
            raise ValueError("Invalid policy output provided")
        #print("exp_adv_shape",exp_adv.shape)
        #print("bc_loss_shape",bc_loss.shape)
        actor_loss = (exp_adv * bc_loss)[:exp_adv.shape[0]//2].mean()

        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        actor_grad_norm = sync_grads(self.actor)
        self.actor_optim.step()

        # update the crtic_network
        self.critic_optim.zero_grad()
        q_loss.backward()
        value_grad_norm = sync_grads(self.critic)
        self.critic_optim.step()
        
        # update the value_network
        self.value_optim.zero_grad()
        v_loss.backward()
        value_grad_norm = sync_grads(self.value)
        self.value_optim.step()

        return actor_loss.item(), q_loss.item(), v_loss.item()
    
    def learn(self):
        if MPI.COMM_WORLD.Get_rank() == 0:
            t0 = time.time()
            stats = {
                'successes': [],
                'hitting_times': [],
                'actor_losses': [],
                'q_losses': [],
                'value_losses': [],
            }

        # put something to the buffer first
        self.prefill_buffer()

        for epoch in range(self.args.n_epochs):
            AL, CL, VL = [], [], []

            for _ in range(self.args.n_cycles):
                (S, A, AG, G), success = self.collect_rollout()
                self.buffer.store_episode(S, A, AG, G)
                self._update_normalizer(S, A, AG, G)
                for _ in range(self.args.n_batches):
                    a_loss, c_loss, v_loss = self._update()
                    AL.append(a_loss); CL.append(c_loss)
                    VL.append(v_loss)

                self._soft_update(self.actor_target, self.actor)
                self._soft_update(self.value_target, self. value)

            global_success_rate, global_hitting_time = self.eval_agent()

            if MPI.COMM_WORLD.Get_rank() == 0:
                t1 = time.time()
                AL = np.array(AL); CL = np.array(CL)
                VL = np.array(VL)
                stats['successes'].append(global_success_rate)
                stats['hitting_times'].append(global_hitting_time)
                stats['actor_losses'].append(AL.mean())
                stats['q_losses'].append(CL.mean())
                stats['value_losses'].append(VL.mean())
                print(f"[info] epoch {epoch:3d} success rate {global_success_rate:6.4f} | "+\
                        f" actor loss {AL.mean():6.4f} |  q loss {CL.mean():6.4f} | "+\
                        f" value loss {VL.mean():6.4f} | "+\
                        f"time {(t1-t0)/60:6.4f} min")
                self.save_model(stats)
