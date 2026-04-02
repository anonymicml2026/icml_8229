import copy
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from mpi4py import MPI
from src.gcrl_model import *
from src.replay_buffer import ReplayBuffer
from src.goal_utils import *
from src.sampler import Sampler
from src.agent.base import Agent


class PPO_HER(Agent):
    """
    Proximal Policy Optimization with Hindsight Experience Replay
    """
    def __init__(self, args, env):
        super().__init__(args, env)
        
        # PPO-specific hyperparameters
        self.clip_epsilon = 0.2  # PPO clipping parameter
        self.value_coef = 0.5    # Value loss coefficient
        self.entropy_coef = 0.01  # Entropy bonus coefficient
        self.max_grad_norm = 0.5  # Gradient clipping
        self.ppo_epochs = 10      # Number of PPO epochs per update
        self.mini_batch_size = 256  # Mini-batch size for PPO updates
        
        # Actor (Gaussian policy)
        self.actor = GaussianActor(args)
        sync_networks(self.actor)
        
        # Value function (Critic)
        critic_map = {
            'monolithic': CriticMonolithic,
        }
        self.critic = critic_map[args.critic](args)
        sync_networks(self.critic)
        
        if self.args.cuda:
            self.actor.cuda()
            self.critic.cuda()
        
        # Optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), 
                                           lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(),
                                            lr=self.args.lr_critic)
        
        # Replay buffer with HER sampling
        self.sample_func = self.sampler.sample_her_transitions
        self.buffer = ReplayBuffer(args, self.sample_func)
        
        # PPO trajectory buffer (for on-policy data)
        self.ppo_buffer = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'goals': [],
            'dones': [],
        }
    
    def select_action(self, o, stochastic=True):
        """
        Select action using the current policy
        Returns action, log_prob, and value for PPO
        """
        s = o['observation'].astype(np.float32)
        g = o['desired_goal'].astype(np.float32)
        s_tensor, g_tensor = self._preproc_inputs(s, g, unsqueeze=True)
        
        with torch.no_grad():
            action, log_prob, _ = self.actor.sample(s_tensor, g_tensor)
            action = action.cpu().numpy().squeeze()
            log_prob = log_prob.cpu().numpy().squeeze()
            
            # Get value estimate
            value = self.critic(s_tensor, 
                              torch.from_numpy(action).float().cuda().unsqueeze(0) if self.args.cuda 
                              else torch.from_numpy(action).float().unsqueeze(0), 
                              g_tensor)
            value = value.cpu().numpy().squeeze()
        
        if not stochastic:
            # For evaluation, use mean of the distribution
            with torch.no_grad():
                action = self.actor.get_action(s_tensor, g_tensor, deterministic=True)
                action = action.cpu().numpy().squeeze()
                return action
        
        # Add exploration noise for training
        max_action = self.args.max_action
        action = np.clip(action, -max_action, max_action)
        
        return action, log_prob, value
    
    def collect_rollout(self, uniform_random_action=False, stochastic=True):
        """
        Collect rollouts and store in both HER buffer and PPO buffer
        """
        n_episodes = self.args.rollout_n_episodes
        dim_state = self.args.dim_state
        dim_action = self.args.dim_action
        dim_goal = self.args.dim_goal
        T = self.args.max_episode_steps
        max_action = self.args.max_action
        
        S = np.zeros((n_episodes, T+1, dim_state), np.float32)
        A = np.zeros((n_episodes, T, dim_action), np.float32)
        AG = np.zeros((n_episodes, T+1, dim_goal), np.float32)
        G = np.zeros((n_episodes, T, dim_goal), np.float32)
        success = np.zeros((n_episodes), np.float32)
        
        # PPO-specific storage
        log_probs = np.zeros((n_episodes, T), np.float32)
        values = np.zeros((n_episodes, T), np.float32)
        rewards = np.zeros((n_episodes, T), np.float32)
        
        for i in range(n_episodes):
            o = self.env.reset()
            for t in range(T):
                if uniform_random_action:
                    a = np.random.uniform(low=-max_action,
                                        high=max_action,
                                        size=(dim_action,))
                    log_prob = 0.0
                    value = 0.0
                else:
                    a, log_prob, value = self.select_action(o, stochastic=stochastic)
                
                new_o, r, d, info = self.env.step(a)
                
                S[i][t] = o['observation'].copy()
                AG[i][t] = o['achieved_goal'].copy()
                G[i][t] = o['desired_goal'].copy()
                A[i][t] = a.copy()
                log_probs[i][t] = log_prob
                values[i][t] = value
                rewards[i][t] = r
                
                o = new_o
            
            success[i] = info['is_success']
            S[i][t+1] = o['observation'].copy()
            AG[i][t+1] = o['achieved_goal'].copy()
        
        # Store in PPO buffer for on-policy updates
        if not uniform_random_action:
            self.ppo_buffer['states'].append(S)
            self.ppo_buffer['actions'].append(A)
            self.ppo_buffer['log_probs'].append(log_probs)
            self.ppo_buffer['values'].append(values)
            self.ppo_buffer['rewards'].append(rewards)
            self.ppo_buffer['goals'].append(G)
        
        return (S, A, AG, G), success
    
    def compute_gae(self, rewards, values, gamma=0.98, lam=0.95):
        """
        Compute Generalized Advantage Estimation (GAE)
        """
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + values
        
        return advantages, returns
    
    def _update_ppo(self):
        """
        Perform PPO update using collected on-policy data
        """
        if len(self.ppo_buffer['states']) == 0:
            return 0.0, 0.0, 0.0
        
        # Concatenate all episodes
        states = np.concatenate(self.ppo_buffer['states'], axis=0)
        actions = np.concatenate(self.ppo_buffer['actions'], axis=0)
        old_log_probs = np.concatenate(self.ppo_buffer['log_probs'], axis=0)
        values = np.concatenate(self.ppo_buffer['values'], axis=0)
        rewards = np.concatenate(self.ppo_buffer['rewards'], axis=0)
        goals = np.concatenate(self.ppo_buffer['goals'], axis=0)
        
        # Reshape to (n_episodes * T, dim)
        n_episodes, T = states.shape[0], states.shape[1] - 1
        states = states[:, :-1].reshape(-1, states.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])
        old_log_probs = old_log_probs.reshape(-1)
        values = values.reshape(-1)
        rewards = rewards.reshape(-1)
        goals = goals.reshape(-1, goals.shape[-1])
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, 
                                               gamma=self.args.gamma, 
                                               lam=0.95)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states, goals = self._preproc_inputs(states, goals)
        actions = numpy2torch(actions, unsqueeze=False, cuda=self.args.cuda)
        old_log_probs = numpy2torch(old_log_probs, unsqueeze=False, cuda=self.args.cuda)
        advantages = numpy2torch(advantages, unsqueeze=False, cuda=self.args.cuda)
        returns = numpy2torch(returns, unsqueeze=False, cuda=self.args.cuda)
        
        # PPO update
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        dataset_size = states.shape[0]
        
        for _ in range(self.ppo_epochs):
            # Shuffle data
            indices = np.random.permutation(dataset_size)
            
            for start_idx in range(0, dataset_size, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, dataset_size)
                mb_indices = indices[start_idx:end_idx]
                
                mb_states = states[mb_indices]
                mb_goals = goals[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # Get current policy predictions
                new_log_probs, entropy = self.actor.evaluate(mb_states, mb_goals, mb_actions)
                
                # Get current value predictions
                new_values = self.critic(mb_states, mb_actions, mb_goals).squeeze(-1)
                
                # Compute policy loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 
                                  1.0 + self.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                critic_loss = F.mse_loss(new_values, mb_returns)
                
                # Compute entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
                
                # Update networks
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                sync_grads(self.actor)
                sync_grads(self.critic)
                
                self.actor_optim.step()
                self.critic_optim.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        # Clear PPO buffer
        self.ppo_buffer = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'goals': [],
            'dones': [],
        }
        
        if n_updates > 0:
            return total_actor_loss / n_updates, total_critic_loss / n_updates, total_entropy / n_updates
        else:
            return 0.0, 0.0, 0.0
    
    def _update_her(self):
        """
        Perform off-policy update using HER replayed data
        Similar to DDPG but with PPO's value function
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
        
        # Compute target value
        with torch.no_grad():
            next_action, _, _ = self.actor.sample(NS, G)
            next_value = self.critic(NS, next_action, G).squeeze(-1).unsqueeze(-1)
            
            clip_return = 1 / (1 - self.args.gamma)
            if self.args.negative_reward:
                target_value = (R + self.args.gamma * next_value).clamp_(-clip_return, 0)
            else:
                target_value = (R + self.args.gamma * next_value).clamp_(0, clip_return)
        
        # Compute current value
        current_value = self.critic(S, A, G)
        
        # Critic loss
        critic_loss = F.mse_loss(current_value, target_value)
        
        # Update critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic)
        self.critic_optim.step()
        
        # Actor loss (policy gradient with value function)
        action_pred, log_prob, _ = self.actor.sample(S, G)
        value_pred = self.critic(S, action_pred, G).squeeze(-1)
        actor_loss = -value_pred.mean()
        
        # Update actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor)
        self.actor_optim.step()
        
        return actor_loss.item(), critic_loss.item()
    
    def learn(self):
        """
        Main training loop combining PPO and HER
        """
        if MPI.COMM_WORLD.Get_rank() == 0:
            t0 = time.time()
            stats = {
                'successes': [],
                'hitting_times': [],
                'ppo_actor_losses': [],
                'ppo_critic_losses': [],
                'ppo_entropy': [],
                'her_actor_losses': [],
                'her_critic_losses': [],
            }
        
        # Prefill buffer
        self.prefill_buffer()
        
        for epoch in range(self.args.n_epochs):
            PPO_AL, PPO_CL, PPO_ENT = [], [], []
            HER_AL, HER_CL = [], []
            
            for _ in range(self.args.n_cycles):
                # Collect rollouts
                (S, A, AG, G), success = self.collect_rollout()
                self.buffer.store_episode(S, A, AG, G)
                self._update_normalizer(S, A, AG, G)
                
                # PPO update (on-policy)
                ppo_a_loss, ppo_c_loss, ppo_ent = self._update_ppo()
                if ppo_a_loss != 0.0:
                    PPO_AL.append(ppo_a_loss)
                    PPO_CL.append(ppo_c_loss)
                    PPO_ENT.append(ppo_ent)
                
                # HER update (off-policy)
                for _ in range(self.args.n_batches):
                    her_a_loss, her_c_loss = self._update_her()
                    HER_AL.append(her_a_loss)
                    HER_CL.append(her_c_loss)
            
            # Evaluate agent
            global_success_rate, global_hitting_time = self.eval_agent()
            
            if MPI.COMM_WORLD.Get_rank() == 0:
                t1 = time.time()
                
                stats['successes'].append(global_success_rate)
                stats['hitting_times'].append(global_hitting_time)
                
                if len(PPO_AL) > 0:
                    PPO_AL = np.array(PPO_AL)
                    PPO_CL = np.array(PPO_CL)
                    PPO_ENT = np.array(PPO_ENT)
                    stats['ppo_actor_losses'].append(PPO_AL.mean())
                    stats['ppo_critic_losses'].append(PPO_CL.mean())
                    stats['ppo_entropy'].append(PPO_ENT.mean())
                else:
                    stats['ppo_actor_losses'].append(0.0)
                    stats['ppo_critic_losses'].append(0.0)
                    stats['ppo_entropy'].append(0.0)
                
                if len(HER_AL) > 0:
                    HER_AL = np.array(HER_AL)
                    HER_CL = np.array(HER_CL)
                    stats['her_actor_losses'].append(HER_AL.mean())
                    stats['her_critic_losses'].append(HER_CL.mean())
                else:
                    stats['her_actor_losses'].append(0.0)
                    stats['her_critic_losses'].append(0.0)
                
                print(f"[info] epoch {epoch:3d} | success rate {global_success_rate:6.4f} | " +
                      f"PPO actor {stats['ppo_actor_losses'][-1]:6.4f} | " +
                      f"PPO critic {stats['ppo_critic_losses'][-1]:6.4f} | " +
                      f"PPO entropy {stats['ppo_entropy'][-1]:6.4f} | " +
                      f"HER actor {stats['her_actor_losses'][-1]:6.4f} | " +
                      f"HER critic {stats['her_critic_losses'][-1]:6.4f} | " +
                      f"time {(t1-t0)/60:6.4f} min")
                
                self.save_model(stats)