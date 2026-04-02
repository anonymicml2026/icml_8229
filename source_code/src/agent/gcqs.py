import copy
import numpy as np
import time
import torch
import threading

from src.gcrl_model import *
from src.replay_buffer import ReplayBuffer
from src.goal_utils import *
from src.agent.base import Agent
import tensorflow as tf


class GCQS(Agent):
    def __init__(self, args, env):
        super().__init__(args, env)
        self.args = args
        self.args.gcqs = True
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.do_mcac_bonus = False
        
        # Actor
        self.actor = GaussianActor(self.args).to(self.device)
        sync_networks(self.actor)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_actor)
        self.actor_target = GaussianActor(self.args).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic
        self.critic 		= EnsembleCritic(self.args).to(self.device)
        sync_networks(self.critic)
        self.critic_target 	= EnsembleCritic(self.args).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_critic)

        self.epsilon = 1e-16
        self.Lambda = 0.1
    
        self.lock = threading.Lock()

        self.sample_gcqs_transitions =  self.sampler.sample_gcqs_transitions

        def sample_func(S, A, AG,  G, size):
            return self.sample_gcqs_transitions(
                    S, A, AG, G, size)
        self.sample_func = sample_func

        # Configure the replay buffer.
        self.buffer = ReplayBuffer(self.args, self.sample_func)

    def value(self, state, goal):
        action,_, _  = self.actor.sample(state, goal)
        V = self.critic(state, action, goal).min(-1, keepdim=True)[0]
        return V
    
    def train_highlevel_policy(self, A,S,G):
        # Compute subgoal distribution 
        A_,_,_ = self.actor.sample(S, G)
        Q = self.critic(S, A_, G)
        Q = torch.min(Q, -1, keepdim=True)[0]
        actor_loss = ((A_ - A).pow(2)- Q).mean()
        #actor_loss += self.args.action_l2 * (A_ / self.args.max_action).pow(2).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor)
        self.actor_optim.step()
        #print("subgoal_loss:",subgoal_loss.item())
            
    def get_action_and_KL(self,S,SG,G):
        B, T = G.shape[:2]
        batch_size = B
        #Sample action, subgoals and KL-divergence
        action_dist = self.actor(S,G)
        action = action_dist.rsample()
      
        with torch.no_grad():
            subgoal = SG

        prior_action_dist = self.actor_target(S.unsqueeze(1).expand(batch_size, subgoal.size(1), self.args.dim_state), subgoal)
        prior_prob = prior_action_dist.log_prob(action.unsqueeze(1).expand(batch_size, subgoal.size(1), self.args.dim_action)).sum(-1, keepdim=True).exp()
        prior_log_prob = torch.log(prior_prob.mean(1) + self.epsilon)
        D_KL = action_dist.log_prob(action).sum(-1, keepdim=True) - prior_log_prob
        
        action = torch.tanh(action)
        return action,D_KL

    def _update(self):
        transition = self.buffer.sample(self.args.batch_size)
        S   = transition['S']
        NS  = transition['NS']
        A   = transition['A']
        G   = transition['G']
        NG = transition['NG']
        R   = transition['R']
        AG = transition['AG']
        DG = transition['DG']
        # S/NS: (batch, dim_state)
        # A: (batch, dim_action)
        # G: (batch, dim_goal)
        A  = numpy2torch(A, unsqueeze=False, cuda=self.args.cuda)
        R  = numpy2torch(R, unsqueeze=False, cuda=self.args.cuda)
        AG = numpy2torch(AG, unsqueeze=False, cuda=self.args.cuda)
        DG = numpy2torch(DG, unsqueeze=False, cuda=self.args.cuda)
        S,   G = self._preproc_inputs(S, G)
        NS,  _ = self._preproc_inputs(NS)
        
        # 1. update critic
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            clip_return = 1 / (1 - self.args.gamma)
            NA,_,_ = self.actor_target.sample(NS, G)
            NQ = self.critic_target(NS, NA, G).detach()
            NQ = torch.min(NQ, -1, keepdim=True)[0]
            
            # TD return
            clip_return = 1 / (1 - self.args.gamma)
            if self.args.negative_reward:
                target = (R + self.args.gamma * NQ).detach().clamp_(-clip_return, 0)
            else:
                target = (R + self.args.gamma * NQ).detach().clamp_(0, clip_return)
        
        Q = self.critic(S, A, G)
        critic_loss = 0.5 * (Q - target).pow(2).sum(-1).mean()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic)
        self.critic_optim.step()
        #print("dual_alpha:",dual_alpha.detach())
        #critic_loss = dual_alpha.detach().cuda() *one_step_critic_loss + (1-dual_alpha.detach().cuda())*n_step_critic_loss

        """ High-level policy learning """
        self.train_highlevel_policy(A,S,G)

        #dual_alpha = self.alpha.param.exp()
        # 2. update actor
        A_,D_KL = self.get_action_and_KL(S,AG,DG)
        Q = self.critic(S, A_, DG)
        Q = torch.min(Q, -1, keepdim=True)[0]
        actor_loss = (self.args.beta*D_KL - Q).mean()
        
        #actor_loss = - self.critic(S, A_, G).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor)
        self.actor_optim.step()
        
        return actor_loss.item(), critic_loss.item()

    def learn(self):
        if MPI.COMM_WORLD.Get_rank() == 0:
            t0 = time.time()
            stats = {
                'successes': [],
                'hitting_times': [],
                'actor_losses': [],
                'critic_losses': [],
            }

        # put something to the buffer first
        self.prefill_buffer()
        if self.args.cuda:
            n_scales = (self.args.max_episode_steps * self.args.rollout_n_episodes // (self.args.n_batches*2)) + 1
        else:
            n_scales = 1
        for epoch in range(self.args.n_epochs):
            AL, CL, DL = [], [], []
            for _ in range(self.args.n_cycles):
                (S, A, AG, G), success = self.collect_rollout()
                self.buffer.store_episode(S, A, AG, G)
                self._update_normalizer(S, A, AG,G)
                for _ in range(n_scales): # scale up for single thread
                    for _ in range(self.args.n_batches):
                        a_loss, c_loss= self._update()
                        AL.append(a_loss); CL.append(c_loss)

                    self._soft_update(self.actor_target, self.actor)
                    self._soft_update(self.critic_target, self.critic)

            global_success_rate, global_hitting_time = self.eval_agent()

            if MPI.COMM_WORLD.Get_rank() == 0:
                t1 = time.time()
                AL = np.array(AL); CL = np.array(CL)
                stats['successes'].append(global_success_rate)
                stats['hitting_times'].append(global_hitting_time)
                stats['actor_losses'].append(AL.mean())
                stats['critic_losses'].append(CL.mean())
                print(f"[info] epoch {epoch:3d} success rate {global_success_rate:6.4f} | "+\
                        f" actor loss {AL.mean():6.4f} | critic loss {CL.mean():6.4f} | "+\
                        f"time {(t1-t0)/60:6.4f} min")
                self.save_model(stats)