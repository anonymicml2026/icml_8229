import copy
import numpy as np
import time
import torch
import torch.nn.functional as F

from mpi4py import MPI
from src.gcrl_model import *
from src.replay_buffer import ReplayBuffer
from src.goal_utils import *
from src.sampler import Sampler
from src.agent.base import Agent


class TD3_HER(Agent):
    """
    Twin Delayed Deep Deterministic Policy Gradient with Hindsight Experience Replay
    
    TD3 improvements over DDPG:
    1. Twin Q-networks (clipped double Q-learning)
    2. Delayed policy updates
    3. Target policy smoothing
    """
    def __init__(self, args, env):
        super().__init__(args, env)
        
        # TD3-specific hyperparameters
        self.policy_delay = 2          # Delayed policy update frequency
        self.target_noise = 0.2        # Target policy smoothing noise
        self.noise_clip = 0.5          # Target policy smoothing noise clip
        self.policy_update_counter = 0  # Counter for delayed policy updates
        
        # Actor network
        self.actor = Actor(args)
        sync_networks(self.actor)
        self.actor_target = copy.deepcopy(self.actor)
        
        if self.args.cuda:
            self.actor.cuda()
            self.actor_target.cuda()
        
        self.actor_optim = torch.optim.Adam(self.actor.parameters(),
                                           lr=self.args.lr_actor)
        
        # Twin Critic networks (Q1 and Q2)
        critic_map = {
            'monolithic': CriticMonolithic,
        }
        self.critic_name = args.critic
        
        # First critic network
        self.critic_1 = critic_map[args.critic](args)
        sync_networks(self.critic_1)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        
        # Second critic network (twin)
        self.critic_2 = critic_map[args.critic](args)
        sync_networks(self.critic_2)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        
        if self.args.cuda:
            self.critic_1.cuda()
            self.critic_1_target.cuda()
            self.critic_2.cuda()
            self.critic_2_target.cuda()
        
        # Separate optimizers for each critic
        self.critic_1_optim = torch.optim.Adam(self.critic_1.parameters(),
                                              lr=self.args.lr_critic)
        self.critic_2_optim = torch.optim.Adam(self.critic_2.parameters(),
                                              lr=self.args.lr_critic)
        
        # Count total parameters
        num_param_critic1 = sum([p.numel() for p in self.critic_1.parameters()])
        num_param_critic2 = sum([p.numel() for p in self.critic_2.parameters()])
        num_param_actor = sum([p.numel() for p in self.actor.parameters()])
        print(f"[info] num parameters - critic_1: {num_param_critic1}, " +
              f"critic_2: {num_param_critic2}, actor: {num_param_actor}")
        
        # Replay buffer with HER
        self.sample_func = self.sampler.sample_her_transitions
        self.buffer = ReplayBuffer(args, self.sample_func)
    
    def _update_critic(self, S, A, G, R, NS, NG):
        """
        Update twin critic networks using TD3's clipped double Q-learning
        """
        with torch.no_grad():
            # Select next action with target policy
            next_action = self.actor_target(NS, G)
            
            # Target policy smoothing: Add clipped noise to target action
            noise = (torch.randn_like(next_action) * self.target_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (next_action + noise).clamp(-self.args.max_action, self.args.max_action)
            
            # Compute target Q-values using both target critics
            target_Q1 = self.critic_1_target(NS, next_action, G)
            target_Q2 = self.critic_2_target(NS, next_action, G)
            
            # Take minimum of twin Q-values (clipped double Q-learning)
            target_Q = torch.min(target_Q1, target_Q2)
            
            # Compute TD target
            clip_return = 1 / (1 - self.args.gamma)
            if self.args.negative_reward:
                target_Q = (R + self.args.gamma * target_Q).clamp_(-clip_return, 0)
                if self.args.terminate:
                    target_Q = target_Q * (-R)
            else:
                target_Q = (R + self.args.gamma * target_Q).clamp_(0, clip_return)
                if self.args.terminate:
                    target_Q = (1 - R) * target_Q + R
        
        # Get current Q estimates from both critics
        current_Q1 = self.critic_1(S, A, G)
        current_Q2 = self.critic_2(S, A, G)
        
        # Compute critic losses
        critic_1_loss = F.mse_loss(current_Q1, target_Q)
        critic_2_loss = F.mse_loss(current_Q2, target_Q)
        
        # Update first critic
        self.critic_1_optim.zero_grad()
        (critic_1_loss * self.args.loss_scale).backward()
        critic_1_grad_norm = sync_grads(self.critic_1)
        self.critic_1_optim.step()
        
        # Update second critic
        self.critic_2_optim.zero_grad()
        (critic_2_loss * self.args.loss_scale).backward()
        critic_2_grad_norm = sync_grads(self.critic_2)
        self.critic_2_optim.step()
        
        return critic_1_loss.item(), critic_2_loss.item(), critic_1_grad_norm, critic_2_grad_norm
    
    def _update_actor(self, S, G):
        """
        Update actor network (delayed policy update)
        """
        # Compute actor loss using only first critic (TD3 uses only Q1 for actor update)
        action = self.actor(S, G)
        actor_loss = -self.critic_1(S, action, G).mean()
        actor_loss += self.args.action_l2 * (action / self.args.max_action).pow(2).mean()
        
        # Update actor
        self.actor_optim.zero_grad()
        (actor_loss * self.args.loss_scale).backward()
        actor_grad_norm = sync_grads(self.actor)
        self.actor_optim.step()
        
        return actor_loss.item(), actor_grad_norm
    
    def _update(self):
        """
        Perform one TD3 update step
        """
        transition = self.buffer.sample(self.args.batch_size)
        
        S = transition['S']
        NS = transition['NS']
        A = transition['A']
        G = transition['G']
        R = transition['R']
        NG = transition['NG']
        
        # Convert to tensors
        A = numpy2torch(A, unsqueeze=False, cuda=self.args.cuda)
        R = numpy2torch(R, unsqueeze=False, cuda=self.args.cuda)
        S, G = self._preproc_inputs(S, G)
        NS, NG = self._preproc_inputs(NS, NG)
        
        # Update critics
        c1_loss, c2_loss, c1_grad_norm, c2_grad_norm = self._update_critic(S, A, G, R, NS, NG)
        
        # Delayed policy update
        self.policy_update_counter += 1
        
        if self.policy_update_counter % self.policy_delay == 0:
            # Update actor
            a_loss, a_grad_norm = self._update_actor(S, G)
            
            # Soft update target networks
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_1_target, self.critic_1)
            self._soft_update(self.critic_2_target, self.critic_2)
            
            return a_loss, c1_loss, c2_loss, a_grad_norm, c1_grad_norm, c2_grad_norm
        else:
            # Return None for actor metrics when not updating
            return None, c1_loss, c2_loss, None, c1_grad_norm, c2_grad_norm
    
    def learn(self):
        """
        Main TD3+HER training loop
        """
        if MPI.COMM_WORLD.Get_rank() == 0:
            t0 = time.time()
            stats = {
                'successes': [],
                'hitting_times': [],
                'actor_losses': [],
                'critic_1_losses': [],
                'critic_2_losses': [],
                'actor_grad_norms': [],
                'critic_1_grad_norms': [],
                'critic_2_grad_norms': [],
            }
        
        # Prefill buffer with random exploration
        self.prefill_buffer()
        
        for epoch in range(self.args.n_epochs):
            AL, C1L, C2L = [], [], []
            AGN, C1GN, C2GN = [], [], []
            
            for _ in range(self.args.n_cycles):
                # Collect rollouts
                (S, A, AG, G), success = self.collect_rollout()
                self.buffer.store_episode(S, A, AG, G)
                self._update_normalizer(S, A, AG, G)
                
                # Perform multiple gradient updates
                for _ in range(self.args.n_batches):
                    a_loss, c1_loss, c2_loss, a_gn, c1_gn, c2_gn = self._update()
                    
                    # Only log actor metrics when actor is updated
                    if a_loss is not None:
                        AL.append(a_loss)
                        AGN.append(a_gn)
                    
                    C1L.append(c1_loss)
                    C2L.append(c2_loss)
                    C1GN.append(c1_gn)
                    C2GN.append(c2_gn)
            
            # Evaluate agent
            global_success_rate, global_hitting_time = self.eval_agent()
            
            if MPI.COMM_WORLD.Get_rank() == 0:
                t1 = time.time()
                
                # Convert lists to arrays for statistics
                C1L = np.array(C1L)
                C2L = np.array(C2L)
                C1GN = np.array(C1GN)
                C2GN = np.array(C2GN)
                
                stats['successes'].append(global_success_rate)
                stats['hitting_times'].append(global_hitting_time)
                stats['critic_1_losses'].append(C1L.mean())
                stats['critic_2_losses'].append(C2L.mean())
                stats['critic_1_grad_norms'].append(C1GN.mean())
                stats['critic_2_grad_norms'].append(C2GN.mean())
                
                # Handle actor metrics (may be empty if not updated this epoch)
                if len(AL) > 0:
                    AL = np.array(AL)
                    AGN = np.array(AGN)
                    stats['actor_losses'].append(AL.mean())
                    stats['actor_grad_norms'].append(AGN.mean())
                    actor_loss_str = f"{AL.mean():6.4f}"
                    actor_gn_str = f"{AGN.mean():6.4f}"
                else:
                    stats['actor_losses'].append(0.0)
                    stats['actor_grad_norms'].append(0.0)
                    actor_loss_str = "N/A   "
                    actor_gn_str = "N/A   "
                
                print(f"[info] epoch {epoch:3d} | success rate {global_success_rate:6.4f} | " +
                      f"actor loss {actor_loss_str} | " +
                      f"critic1 loss {C1L.mean():6.4f} | " +
                      f"critic2 loss {C2L.mean():6.4f} | " +
                      f"actor gradnorm {actor_gn_str} | " +
                      f"critic1 gradnorm {C1GN.mean():6.4f} | " +
                      f"critic2 gradnorm {C2GN.mean():6.4f} | " +
                      f"time {(t1-t0)/60:6.4f} min")
                
                self.save_model(stats)
    
    def save_model(self, stats):
        """
        Save model with both critic networks
        """
        sd = {
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "args": self.args,
            "stats": stats,
            "s_mean": self.s_norm.mean,
            "s_std": self.s_norm.std,
            "g_mean": self.g_norm.mean,
            "g_std": self.g_norm.std,
        }
        
        torch.save(sd,
                  os.path.join(self.args.save_dir, f"{self.args.experiment_name}.pt"))
        
        if stats['successes'][-1] > self.best_success_rate:
            self.best_success_rate = stats['successes'][-1]
            torch.save(sd,
                      os.path.join(self.args.save_dir, f"{self.args.experiment_name}_best.pt"))