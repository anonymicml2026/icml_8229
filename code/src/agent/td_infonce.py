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


class TDInfoNCE(Agent):
    """
    TD-InfoNCE agent for goal-conditioned tasks
    Uses contrastive learning with future and random goal relabeling
    """
    def __init__(self, args, env):
        super().__init__(args, env)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor (Gaussian policy for exploration)
        self.actor = GaussianActor(self.args).to(self.device)
        sync_networks(self.actor)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_actor)
        
        # Twin Bilinear Critics for contrastive learning
        self.critic = TwinBilinearCritic(self.args).to(self.device)
        sync_networks(self.critic)
        self.critic_target = TwinBilinearCritic(self.args).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_critic)
        
        self.lock = threading.Lock()
        
        # Sample function for TD-InfoNCE with future and random goal relabeling
        self.sample_func = self.sampler.sample_td_infonce_transitions
        self.buffer = ReplayBuffer(args, self.sample_func)
    
    def _preproc_inputs(self, S, G=None):
        """Preprocess and normalize inputs"""
        S_norm = self.s_norm.normalize(S)
        S_tensor = numpy2torch(S_norm, unsqueeze=False, cuda=self.args.cuda)
        
        if G is not None:
            G_norm = self.g_norm.normalize(G)
            G_tensor = numpy2torch(G_norm, unsqueeze=False, cuda=self.args.cuda)
            return S_tensor, G_tensor
        return S_tensor, None
    
    def _compute_contrastive_critic_loss(self, S, A, G, NS, future_G, random_G):
        """
        Compute TD-InfoNCE contrastive critic loss
        
        Two terms:
        1. Positive pairs: (s,a,g) matched with next_state achieving g
        2. Negative pairs: (s,a,g) contrasted with random goals
        """
        batch_size = S.shape[0]
        
        # Get representations for state-action-goal
        sag_repr, next_state_repr = self.critic(S, A, G, NS, return_repr=True)
        # sag_repr: (batch, repr_dim, 2) for twin critics
        # next_state_repr: (batch, repr_dim, 2)
        
        # Compute logits via inner product (batch_size x batch_size matrix)
        # pos_logits[i,j] = <sag_repr[i], next_state_repr[j]>
        pos_logits = torch.einsum('ikt,jkt->ijk', sag_repr, next_state_repr)
        # pos_logits: (batch, batch, 2)
        
        # Term 1: Positive contrastive loss
        # Target is identity matrix (each sag should match its own next state)
        I = torch.eye(batch_size, device=self.device)
        I = I.unsqueeze(-1).repeat(1, 1, 2)  # (batch, batch, 2)
        
        # Compute softmax cross entropy for each twin critic
        loss1 = F.cross_entropy(
            pos_logits[:, :, 0], 
            torch.arange(batch_size, device=self.device)
        )
        loss1 += F.cross_entropy(
            pos_logits[:, :, 1],
            torch.arange(batch_size, device=self.device)
        )
        loss1 = loss1 / 2.0  # Average over twin critics
        
        # Term 2: Negative contrastive loss with importance sampling
        # Roll random goals to create negatives
        rand_g_repr = self.critic.get_goal_repr(random_G)  # (batch, repr_dim, 2)
        
        neg_logits = torch.einsum('ikt,jkt->ijk', sag_repr, rand_g_repr)
        # neg_logits: (batch, batch, 2)
        
        # Compute importance sampling weights using target network
        with torch.no_grad():
            # Sample next actions
            next_action, _, _ = self.actor.sample(NS, G)
            
            # Get target Q values for weighting
            _, target_next_repr = self.critic_target(NS, next_action, G, NS, return_repr=True)
            logits_w = torch.einsum('ikt,jkt->ijk', target_next_repr, rand_g_repr)
            # logits_w: (batch, batch, 2)
            
            # Min over twin critics
            logits_w = torch.min(logits_w, dim=-1)[0]  # (batch, batch)
            
            # Softmax to get importance weights (per row)
            w = F.softmax(logits_w, dim=1)  # (batch, batch)
        
        # Compute weighted negative loss
        # Expand w for twin critics
        w_expanded = w.unsqueeze(-1).repeat(1, 1, 2)  # (batch, batch, 2)
        
        # Compute cross entropy with importance weights as soft targets
        loss2 = -(w_expanded * F.log_softmax(neg_logits, dim=1)).sum(dim=1).mean()
        
        # Total loss: (1-gamma) * positive + gamma * negative
        critic_loss = (1 - self.args.gamma) * loss1 + self.args.gamma * loss2
        
        # Metrics for logging
        metrics = {
            'critic_loss_pos': loss1.item(),
            'critic_loss_neg': loss2.item(),
            'logits_pos_mean': torch.diagonal(pos_logits[:, :, 0]).mean().item(),
            'logits_neg_mean': neg_logits.mean().item(),
            'importance_weight_diag': torch.diagonal(w).mean().item(),
        }
        
        return critic_loss, metrics
    
    def _update_critic(self, S, A, G, NS, future_G, random_G):
        """Update critic networks"""
        critic_loss, metrics = self._compute_contrastive_critic_loss(
            S, A, G, NS, future_G, random_G
        )
        
        # Optimize the critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        critic_grad_norm = sync_grads(self.critic)
        self.critic_optim.step()
        
        metrics['critic_grad_norm'] = critic_grad_norm
        return critic_loss.item(), metrics
    
    def _update_actor(self, S, G):
        """
        Update actor network
        Actor objective: maximize Q(s, pi(s,g), g, g)
        i.e., learn to reach goal g from state s
        """
        # Sample actions from current policy
        action, log_pi, _ = self.actor.sample(S, G)
        
        batch_size = S.shape[0]
        
        # Compute Q-values where future_goal = current_goal
        # This represents "reaching the goal g"
        goal_repr = self.critic.get_goal_repr(G)  # (batch, repr_dim, 2)
        sag_repr, _ = self.critic(S, action, G, S, return_repr=True)
        
        # Inner product between sag_repr and goal_repr (for same goal)
        logits = torch.einsum('ikt,jkt->ijk', sag_repr, goal_repr)
        # logits: (batch, batch, 2)
        
        # Min over twin critics
        logits = torch.min(logits, dim=-1)[0]  # (batch, batch)
        
        # Target: diagonal (each state-action should reach its own goal)
        I = torch.eye(batch_size, device=self.device)
        
        # Actor loss: cross entropy (want diagonal to be high)
        actor_loss = F.cross_entropy(logits, torch.arange(batch_size, device=self.device))
        
        # Optimize the actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        actor_grad_norm = sync_grads(self.actor)
        self.actor_optim.step()
        
        metrics = {
            'actor_grad_norm': actor_grad_norm,
            'actor_logits_diag': torch.diagonal(logits).mean().item(),
        }
        
        return actor_loss.item(), metrics
    
    def _update(self):
        """Perform one update step"""
        transition = self.buffer.sample(self.args.batch_size)
        
        S = transition['S']
        NS = transition['NS']
        A = transition['A']
        G = transition['G']
        future_G = transition['future_G']
        random_G = transition['random_G']
        
        # Convert to tensors
        A = numpy2torch(A, unsqueeze=False, cuda=self.args.cuda)
        S, G = self._preproc_inputs(S, G)
        NS, _ = self._preproc_inputs(NS)
        future_G_norm = self.g_norm.normalize(future_G)
        future_G_tensor = numpy2torch(future_G_norm, unsqueeze=False, cuda=self.args.cuda)
        random_G_norm = self.g_norm.normalize(random_G)
        random_G_tensor = numpy2torch(random_G_norm, unsqueeze=False, cuda=self.args.cuda)
        
        # Update critic
        critic_loss, critic_metrics = self._update_critic(
            S, A, G, NS, future_G_tensor, random_G_tensor
        )
        
        # Update actor (use future goals as targets for actor)
        actor_loss, actor_metrics = self._update_actor(S, future_G_tensor)
        
        # Soft update target networks
        self._soft_update(self.critic_target, self.critic)
        
        # Combine metrics
        metrics = {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
        }
        metrics.update(critic_metrics)
        metrics.update(actor_metrics)
        
        return actor_loss, critic_loss, metrics.get('actor_grad_norm', 0), metrics.get('critic_grad_norm', 0)

    def learn(self):
        """Main training loop"""
        if MPI.COMM_WORLD.Get_rank() == 0:
            t0 = time.time()
            stats = {
                'successes': [],
                'hitting_times': [],
                'actor_losses': [],
                'critic_losses': [],
                'actor_grad_norms': [],
                'critic_grad_norms': [],
            }

        # Prefill buffer with random exploration
        self.prefill_buffer()
        
        # Calculate update scaling factor
        if self.args.cuda:
            n_scales = (self.args.max_episode_steps * self.args.rollout_n_episodes // (self.args.n_batches * 2)) + 1
        else:
            n_scales = 1
        
        for epoch in range(self.args.n_epochs):
            AL, CL, AGN, CGN = [], [], [], []
            
            for _ in range(self.args.n_cycles):
                # Collect rollout
                (S, A, AG, G), success = self.collect_rollout()
                self.buffer.store_episode(S, A, AG, G)
                self._update_normalizer(S, A, AG, G)
                
                # Update networks
                for _ in range(n_scales):
                    for _ in range(self.args.n_batches):
                        a_loss, c_loss, a_gn, c_gn = self._update()
                        AL.append(a_loss)
                        CL.append(c_loss)
                        AGN.append(a_gn)
                        CGN.append(c_gn)

            # Evaluate agent
            global_success_rate, global_hitting_time = self.eval_agent()

            if MPI.COMM_WORLD.Get_rank() == 0:
                t1 = time.time()
                AL = np.array(AL)
                CL = np.array(CL)
                AGN = np.array(AGN)
                CGN = np.array(CGN)
                
                stats['successes'].append(global_success_rate)
                stats['hitting_times'].append(global_hitting_time)
                stats['actor_losses'].append(AL.mean())
                stats['critic_losses'].append(CL.mean())
                stats['actor_grad_norms'].append(AGN.mean())
                stats['critic_grad_norms'].append(CGN.mean())
                
                print(f"[info] epoch {epoch:3d} | success rate {global_success_rate:6.4f} | " +
                      f"actor loss {AL.mean():6.4f} | critic loss {CL.mean():6.4f} | " +
                      f"actor gradnorms {AGN.mean():6.4f} | critic gradnorms {CGN.mean():6.4f} | " +
                      f"time {(t1-t0)/60:6.4f} min")
                
                self.save_model(stats)