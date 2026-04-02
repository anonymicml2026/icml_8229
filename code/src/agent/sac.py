import copy
import numpy as np
import time
import torch
import torch.nn.functional as F
import threading

from mpi4py import MPI
from src.gcrl_model import *
from src.replay_buffer import ReplayBuffer
from src.goal_utils import *
from src.sampler import Sampler
from src.agent.base import Agent


class SAC(Agent):
    """
    Standard Soft Actor-Critic agent for goal-conditioned tasks
    """
    def __init__(self, args, env):
        super().__init__(args, env)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args.gchr = True
        
        # Actor (Gaussian policy)
        self.actor = GaussianActor(self.args).to(self.device)
        sync_networks(self.actor)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_actor)
        
        # Critic (Double Q-networks)
        self.critic = CriticMonolithic(self.args).to(self.device)
        sync_networks(self.critic)
        self.critic_target = CriticMonolithic(self.args).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_critic)

        # Automatic entropy tuning
        if self.args.target_entropy is None:
            self.target_entropy = -self.args.dim_action
        else:
            self.target_entropy = self.args.target_entropy
        
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.args.alpha_lr)
        
        self.lock = threading.Lock()
        
        # Sample function
        self.sample_func = self.sampler.sample_ddpg_transitions
        self.buffer = ReplayBuffer(args, self.sample_func)
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def _update_critic(self, S, A, G, R, NS):
        """
        Update critic networks
        """
        with torch.no_grad():
            # Sample actions from current policy for next states
            next_action, next_log_pi, _ = self.actor.sample(NS, G)
            
            # Compute target Q-values using double Q-learning
            target_Q = self.critic_target(NS, next_action, G)  # (batch, 2)
            target_Q = torch.min(target_Q, dim=-1, keepdim=True)[0]  # (batch, 1)
            
            # Add entropy term to target
            target_Q = target_Q - self.alpha.detach() * next_log_pi
            
            # Compute TD target
            clip_return = 1 / (1 - self.args.gamma)
            if self.args.negative_reward:
                target_Q = (R + self.args.gamma * target_Q).clamp_(-clip_return, 0)
            else:
                target_Q = (R + self.args.gamma * target_Q).clamp_(0, clip_return)
        
        # Get current Q estimates
        current_Q = self.critic(S, A, G)  # (batch, 2)
        
        # Compute critic loss (MSE for both Q-networks)
        critic_loss = 0.5 * F.mse_loss(current_Q, target_Q.expand(-1, 2))
        
        # Optimize the critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic)
        self.critic_optim.step()
        
        return critic_loss.item()
    
    def _update_actor_and_alpha(self, S, G):
        """
        Update actor network and temperature parameter
        """
        # Sample actions from current policy
        action, log_pi, _ = self.actor.sample(S, G)
        
        # Compute Q-values
        Q = self.critic(S, action, G)  # (batch, 2)
        Q = torch.min(Q, dim=-1, keepdim=True)[0]  # (batch, 1)
        
        # Compute actor loss (maximize Q - alpha * log_pi)
        actor_loss = (self.alpha.detach() * log_pi - Q).mean()
        
        # Optimize the actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor)
        self.actor_optim.step()
        
        # Update temperature parameter
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        
        return actor_loss.item(), alpha_loss.item()
    
    def _update(self):
        """
        Perform one update step
        """
        transition = self.buffer.sample(self.args.batch_size)
        
        S = transition['S']
        NS = transition['NS']
        A = transition['A']
        G = transition['G']
        R = transition['R']
        
        # Convert to tensors
        A = numpy2torch(A, unsqueeze=False, cuda=self.args.cuda)
        R = numpy2torch(R, unsqueeze=False, cuda=self.args.cuda)
        S, G = self._preproc_inputs(S, G)
        NS, _ = self._preproc_inputs(NS)
        
        # Update critic
        critic_loss = self._update_critic(S, A, G, R, NS)
        
        # Update actor and alpha
        actor_loss, alpha_loss = self._update_actor_and_alpha(S, G)
        
        return actor_loss, critic_loss, alpha_loss

    def learn(self):
        """
        Main training loop
        """
        if MPI.COMM_WORLD.Get_rank() == 0:
            t0 = time.time()
            stats = {
                'successes': [],
                'hitting_times': [],
                'actor_losses': [],
                'critic_losses': [],
                'alpha_losses': [],
                'alpha_values': [],
            }

        # Prefill buffer with random exploration
        self.prefill_buffer()
        
        # Calculate update scaling factor
        if self.args.cuda:
            n_scales = (self.args.max_episode_steps * self.args.rollout_n_episodes // (self.args.n_batches * 2)) + 1
        else:
            n_scales = 1
        
        for epoch in range(self.args.n_epochs):
            AL, CL, AlphaL = [], [], []
            
            for _ in range(self.args.n_cycles):
                # Collect rollout
                (S, A, AG, G), success = self.collect_rollout()
                self.buffer.store_episode(S, A, AG, G)
                self._update_normalizer(S, A, AG, G)
                
                # Update networks
                for _ in range(n_scales):
                    for _ in range(self.args.n_batches):
                        a_loss, c_loss, alpha_loss = self._update()
                        AL.append(a_loss)
                        CL.append(c_loss)
                        AlphaL.append(alpha_loss)
                    
                    # Soft update target networks
                    self._soft_update(self.critic_target, self.critic)

            # Evaluate agent
            global_success_rate, global_hitting_time = self.eval_agent()

            if MPI.COMM_WORLD.Get_rank() == 0:
                t1 = time.time()
                AL = np.array(AL)
                CL = np.array(CL)
                AlphaL = np.array(AlphaL)
                
                stats['successes'].append(global_success_rate)
                stats['hitting_times'].append(global_hitting_time)
                stats['actor_losses'].append(AL.mean())
                stats['critic_losses'].append(CL.mean())
                stats['alpha_losses'].append(AlphaL.mean())
                stats['alpha_values'].append(self.alpha.item())
                
                print(f"[info] epoch {epoch:3d} | success rate {global_success_rate:6.4f} | " +
                      f"actor loss {AL.mean():6.4f} | critic loss {CL.mean():6.4f} | " +
                      f"alpha {self.alpha.item():6.4f} | alpha loss {AlphaL.mean():6.4f} | " +
                      f"time {(t1-t0)/60:6.4f} min")
                
                self.save_model(stats)
