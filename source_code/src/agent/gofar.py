import copy
import numpy as np
import time
import torch

from mpi4py import MPI
from src.gcrl_model import *
from src.gofar_replay_buffer import GofarReplayBuffer
from src.goal_utils import *
from src.sampler import Sampler
from src.agent.base import Agent
from src.discriminator import Discriminator


"""
GoFAR (Goal-conditioned f-Advantage Regression)

"""
class GoFar(Agent):
    def __init__(self, args, env):
        super().__init__(args, env) 
        # create the network
        self.args = args
        self.args.relabel_rate = 0
        self.args.reward_type = 'disc' 
        self.args.use_disc = True
        self.args.gofar =True
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor
        self.actor = GofarActor(self.args).to(self.device)
        sync_networks(self.actor)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_actor)
        self.actor_target = GofarActor(self.args).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Value
        self.value 		= Value(self.args).to(self.device)
        sync_networks(self.value)
        self.value_target 	= Value(self.args).to(self.device)
        self.value_target.load_state_dict(self.value.state_dict())
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.args.lr_critic)

        self.discriminator = Discriminator(2 * self.args.dim_goal, lr=args.lr_critic).to(self.device)
        self.buffer = GofarReplayBuffer(args, self.sampler.sample_gofar_transitions)
        print("args.gofar:",args.gofar)
    
    def _deterministic_action(self, input_tensor):
        action = self.actor(input_tensor)
        return action

    def _update_discriminator(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)

        # start to do the update
        ag_norm = self.g_norm.normalize(transitions['AG'])
        g_norm = self.g_norm.normalize(transitions['G'])

        pos_pairs = numpy2torch(np.concatenate([g_norm, g_norm], axis=-1),  unsqueeze=False, cuda=self.args.cuda)
        neg_pairs = numpy2torch(np.concatenate([ag_norm, g_norm], axis=-1), unsqueeze=False, cuda=self.args.cuda)

        expert_d = self.discriminator.trunk(pos_pairs)
        policy_d = self.discriminator.trunk(neg_pairs)

        expert_loss = F.binary_cross_entropy_with_logits(
            expert_d,
            torch.ones(expert_d.size()).to(pos_pairs.device))
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_d,
            torch.zeros(policy_d.size()).to(neg_pairs.device))

        gail_loss = expert_loss + policy_loss
        grad_pen = self.discriminator.compute_grad_pen(pos_pairs, neg_pairs, lambda_=self.args.disc_lambda)

        self.discriminator.optimizer.zero_grad()
        (gail_loss + grad_pen).backward()
        self.discriminator.optimizer.step()

    def _check_discriminator(self):
        transitions = self.buffer.sample(self.args.batch_size)
        ag_norm = self.g_norm.normalize(transitions['AG'])
        g_norm = self.g_norm.normalize(transitions['G'])
        goal_pair = numpy2torch(np.concatenate([g_norm, g_norm], axis=-1),  unsqueeze=False, cuda=self.args.cuda)
        ag_pair = numpy2torch(np.concatenate([ag_norm, ag_norm], axis=-1),  unsqueeze=False, cuda=self.args.cuda)
        diff_pair = numpy2torch(np.concatenate([ag_norm, g_norm], axis=-1),  unsqueeze=False, cuda=self.args.cuda)
        with torch.no_grad():
            goal_pair_score = self.discriminator.predict_reward(goal_pair).mean().cpu().detach().numpy()
            ag_pair_score = self.discriminator.predict_reward(ag_pair).mean().cpu().detach().numpy() 
            ag_g_score = self.discriminator.predict_reward(diff_pair).mean().cpu().detach().numpy()
        #print(f"goal pair: {goal_pair_score:.3f}, ag pair: {ag_pair_score:.3f}, ag-g: {ag_g_score:.3f}")

    # update the network
    def _update(self):
        transition = self.sample_batch()
        IS = transition['IS']
        S  = transition['S']
        NS = transition['NS']
        A  = transition['A']
        AG = transition['AG']
        G  = transition['G']
        R  = transition['R']
        NG = transition['NG']
        NAG = transition['NAG']

        # start to do the update
        io_norm = self.s_norm.normalize(IS)
        obs_norm = self.s_norm.normalize(S)
        ag_norm = self.g_norm.normalize(AG)
        g_norm = self.g_norm.normalize(G)

        inputs_initial_norm = np.concatenate([io_norm, g_norm], axis=-1)
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=-1)
        obs_next_norm = self.s_norm.normalize(NS)
        ag_next_norm = self.g_norm.normalize(NAG)
        g_next_norm = self.g_norm.normalize(NG)
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=-1)
        
        # transfer them into the tensor
        inputs_initial_norm_tensor = numpy2torch(inputs_initial_norm, unsqueeze=False, cuda=self.args.cuda)
        inputs_norm_tensor = numpy2torch(inputs_norm, unsqueeze=False, cuda=self.args.cuda)
        inputs_next_norm_tensor = numpy2torch(inputs_next_norm , unsqueeze=False, cuda=self.args.cuda)
        actions_tensor = numpy2torch(A, unsqueeze=False, cuda=self.args.cuda)
        r_tensor = numpy2torch(R, unsqueeze=False, cuda=self.args.cuda) 
        # r_tensor = - torch.tensor(np.linalg.norm(transitions['ag_next']-transitions['g']), dtype=torch.float32) ** 2

        # obtain discriminator reward
        disc_inputs_norm_tensor = numpy2torch(np.concatenate([ag_norm, g_norm], axis=-1), unsqueeze=False, cuda=self.args.cuda)

        if self.args.reward_type == 'disc':
            #print("hhhh")
            r_tensor = self.discriminator.predict_reward(disc_inputs_norm_tensor)
        elif self.args.reward_type == 'positive':
            r_tensor = r_tensor + 1.
        elif self.args.reward_type == 'square':
            r_tensor = - torch.tensor(np.linalg.norm(ag_next_norm-g_norm, axis=1) ** 2, dtype=torch.float32).unsqueeze(1)
        elif self.args.reward_type == 'laplace':
            r_tensor = - torch.tensor(np.linalg.norm(ag_next_norm-g_norm, ord=1, axis=1) ** 2, dtype=torch.float32).unsqueeze(1)

        # Calculate value loss
        v_initial = self.value(inputs_initial_norm_tensor)
        v_current = self.value(inputs_norm_tensor)
        with torch.no_grad():
            v_next = self.value_target(inputs_next_norm_tensor).detach()
            v_onestep = (r_tensor + self.args.gamma * v_next).detach()

            # if self.args.reward_type == 'binary':
            # v_onestep = torch.clamp(v_onestep, -clip_return, 0)

        # e_v = r_tensor + self.args.gamma * v_next - v_current
        e_v =  v_onestep - v_current 

        v_loss0 = (1 - self.args.gamma) * v_initial 
        if self.args.f == 'chi':
            v_loss1 = torch.mean((e_v + 1).pow(2))
        elif self.args.f == 'kl':
            v_loss1 = torch.log(torch.mean(torch.exp(e_v)))
        value_loss = (v_loss0 + v_loss1).mean()

        # Compute policy loss (L2 because Gaussian with fixed sigma)
        if self.args.f == 'chi':
            w_e = torch.relu(e_v + 1).detach()
        elif self.args.f == 'kl':
            w_e = torch.clamp(torch.exp(e_v.detach()), 0, 10)
        actions_real = self.actor(inputs_norm_tensor)
        actor_loss = torch.mean(w_e * torch.square(actions_real - actions_tensor))

        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        actor_grad_norm = sync_grads(self.actor)
        self.actor_optim.step()

        # update the value_network
        self.value_optim.zero_grad()
        value_loss.backward()
        value_grad_norm = sync_grads(self.value)
        self.value_optim.step()

        return actor_loss.item(), value_loss.item(), actor_grad_norm, value_grad_norm
    
    def learn(self):
        if MPI.COMM_WORLD.Get_rank() == 0:
            t0 = time.time()
            stats = {
                'successes': [],
                'hitting_times': [],
                'actor_losses': [],
                'value_losses': [],
                'actor_grad_norms': [],
                'value_grad_norms': [],
            }

        # put something to the buffer first
        self.prefill_buffer()

        for epoch in range(self.args.n_epochs):
            AL, CL, AGN, CGN = [], [], [], []

            for _ in range(self.args.n_cycles):
                (S, A, AG, G), success = self.collect_rollout()
                self.buffer.store_episode(S, A, AG, G)
                self._update_normalizer(S, A, AG, G)
                # train discriminator
                if self.args.use_disc:
                    for _ in range(self.args.disc_iter):
                        self._update_discriminator()
                for _ in range(self.args.n_batches):
                    a_loss, c_loss, a_gn, c_gn = self._update()
                    AL.append(a_loss); CL.append(c_loss)
                    AGN.append(a_gn); CGN.append(c_gn)

                self._soft_update(self.actor_target, self.actor)
                self._soft_update(self.value_target, self. value)

            global_success_rate, global_hitting_time = self.eval_agent()

            if MPI.COMM_WORLD.Get_rank() == 0:
                t1 = time.time()
                AL = np.array(AL); CL = np.array(CL)
                AGN = np.array(AGN); CGN = np.array(CGN)
                stats['successes'].append(global_success_rate)
                stats['hitting_times'].append(global_hitting_time)
                stats['actor_losses'].append(AL.mean())
                stats['value_losses'].append(CL.mean())
                stats['actor_grad_norms'].append(AGN.mean())
                stats['value_grad_norms'].append(CGN.mean())
                print(f"[info] epoch {epoch:3d} success rate {global_success_rate:6.4f} | "+\
                        f" actor loss {AL.mean():6.4f} |  value loss {CL.mean():6.4f} | "+\
                        f" actor gradnorms {AGN.mean():6.4f} |  value gradnorms {CGN.mean():6.4f} | "+\
                        f"time {(t1-t0)/60:6.4f} min")
                self.save_model(stats)
