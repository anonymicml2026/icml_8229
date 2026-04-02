import copy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from mpi4py import MPI
from src.gcrl_model import Actor
from src.replay_buffer import ReplayBuffer
from src.goal_utils import *
from src.sampler import Sampler
from src.agent.base import Agent


################################################################################
#
# QRL Network Components
#
################################################################################

class Encoder(nn.Module):
    """Encoder network to map observations to latent space"""
    def __init__(self, input_dim: int, latent_dim: int = 128, hidden_dims: List[int] = [512, 512]):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(last_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, latent_dim))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentDynamics(nn.Module):
    """Latent dynamics model: z' = f(z, a)"""
    def __init__(self, latent_dim: int, action_dim: int, hidden_dims: List[int] = [512, 512], residual: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.residual = residual
        
        layers = []
        input_dim = latent_dim + action_dim
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(last_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, latent_dim))
        
        self.net = nn.Sequential(*layers)
        self._init_weights(zero_last=residual)
    
    def _init_weights(self, zero_last: bool = False):
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Linear):
                if zero_last and i == len(list(self.modules())) - 1:
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        za = torch.cat([z, a], dim=-1)
        delta_z = self.net(za)
        if self.residual:
            return z + delta_z
        return delta_z


class IQEQuasimetric(nn.Module):
    """Interval Quasimetric Embedding (IQE) head"""
    def __init__(self, input_dim: int, num_components: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.num_components = num_components
        assert input_dim % num_components == 0, "input_dim must be divisible by num_components"
        self.component_dim = input_dim // num_components
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute quasimetric distance d(x, y)
        x, y: (batch, input_dim)
        returns: (batch,)
        """
        # Reshape to components
        x = x.view(*x.shape[:-1], self.num_components, self.component_dim)
        y = y.view(*y.shape[:-1], self.num_components, self.component_dim)
        
        # Compute max over each component dimension
        # d_c = max(0, max_i(x_ci - y_ci))
        diff = x - y  # (batch, num_components, component_dim)
        max_diff_per_component = diff.max(dim=-1).values  # (batch, num_components)
        max_diff_per_component = torch.relu(max_diff_per_component)
        
        # Sum over components
        dist = max_diff_per_component.sum(dim=-1)  # (batch,)
        return dist


class QuasimetricModel(nn.Module):
    """Quasimetric model: projects latents and computes quasimetric distance"""
    def __init__(self, latent_dim: int, projection_dim: int = 2048, 
                 num_components: int = 64, projector_hidden: List[int] = [512]):
        super().__init__()
        self.latent_dim = latent_dim
        self.projection_dim = projection_dim
        
        # Projector MLP
        layers = []
        last_dim = latent_dim
        for hidden_dim in projector_hidden:
            layers.extend([
                nn.Linear(last_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, projection_dim))
        
        self.projector = nn.Sequential(*layers)
        self.quasimetric_head = IQEQuasimetric(projection_dim, num_components)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, zx: torch.Tensor, zy: torch.Tensor, bidirectional: bool = False) -> torch.Tensor:
        """
        Compute quasimetric distance d(zx, zy)
        If bidirectional=True, returns (d(zx,zy), d(zy,zx))
        """
        px = self.projector(zx)
        py = self.projector(zy)
        
        if bidirectional:
            dist_xy = self.quasimetric_head(px, py)
            dist_yx = self.quasimetric_head(py, px)
            return torch.stack([dist_xy, dist_yx], dim=-1)
        else:
            return self.quasimetric_head(px, py)


class QuasimetricCritic(nn.Module):
    """Complete quasimetric critic: encoder + dynamics + quasimetric"""
    def __init__(self, args, latent_dim: int = 128):
        super().__init__()
        self.args = args
        dim_state = args.dim_state
        dim_action = args.dim_action
        dim_goal = args.dim_goal
        
        self.latent_dim = latent_dim
        
        # Encoder for state (observation)
        self.encoder = Encoder(dim_state + dim_goal, latent_dim)
        
        # Latent dynamics
        self.latent_dynamics = LatentDynamics(latent_dim, dim_action)
        
        # Quasimetric model
        self.quasimetric_model = QuasimetricModel(latent_dim)
    
    def forward(self, s: torch.Tensor, a: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Compute Q(s, a, g) as negative distance to goal
        """
        # Encode current state and goal
        sg = torch.cat([s, g], dim=-1)
        zs = self.encoder(sg)
        
        # Predict next latent
        zs_next = self.latent_dynamics(zs, a)
        
        # Encode goal
        gg = torch.cat([g, g], dim=-1)  # goal as both state and goal
        zg = self.encoder(gg)
        
        # Compute distance
        dist = self.quasimetric_model(zs_next, zg)
        
        # Q-value is negative distance
        return -dist


################################################################################
#
# QRL Agent
#
################################################################################

class QRL(Agent):
    """
    Quasimetric Reinforcement Learning (QRL) agent for goal-conditioned tasks
    """
    def __init__(self, args, env):
        super().__init__(args, env)
        self.args = args
        
        # QRL hyperparameters
        self.latent_dim = 128
        self.num_critics = getattr(args, 'num_critics', 2)
        
        # Loss weights
        self.local_constraint_epsilon = 0.25
        self.local_constraint_step_cost = 1.0
        self.latent_dynamics_weight = 0.1
        self.global_push_weight = 1.0
        self.global_push_beta = 0.1
        self.global_push_offset = 15.0
        
        # Actor (use standard actor from base)
        self.actor = Actor(args)
        sync_networks(self.actor)
        
        # Create ensemble of critics
        self.critics = nn.ModuleList([
            QuasimetricCritic(args, self.latent_dim) 
            for _ in range(self.num_critics)
        ])
        for critic in self.critics:
            sync_networks(critic)
        
        # Target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critics_target = nn.ModuleList([
            copy.deepcopy(critic) for critic in self.critics
        ])
        
        if self.args.cuda:
            self.actor.cuda()
            self.actor_target.cuda()
            self.critics.cuda()
            self.critics_target.cuda()
        
        # Optimizers
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), 
            lr=self.args.lr_actor
        )
        self.critic_optims = [
            torch.optim.Adam(critic.parameters(), lr=self.args.lr_critic)
            for critic in self.critics
        ]
        
        # Lagrange multipliers for local constraint (one per critic)
        self.raw_lagrange_mults = nn.ParameterList([
            nn.Parameter(torch.tensor(np.log(np.expm1(0.01)), dtype=torch.float32))
            for _ in range(self.num_critics)
        ])
        self.lagrange_optims = [
            torch.optim.Adam([mult], lr=self.args.lr_critic)
            for mult in self.raw_lagrange_mults
        ]
        
        if self.args.cuda:
            self.raw_lagrange_mults.cuda()
        
        # Sample function
        self.sample_func = self.sampler.sample_her_transitions
        self.buffer = ReplayBuffer(args, self.sample_func)
    
    def _compute_critic_losses(self, S, A, G, NS, NG):
        """Compute losses for all critics"""
        losses_info = {}
        
        for idx, (critic, critic_target, critic_optim, lagrange_mult, lagrange_optim) in enumerate(
            zip(self.critics, self.critics_target, self.critic_optims, 
                self.raw_lagrange_mults, self.lagrange_optims)
        ):
            # Encode states
            SG = torch.cat([S, G], dim=-1)
            NSG = torch.cat([NS, NG], dim=-1)
            zx = critic.encoder(SG)
            zy = critic.encoder(NSG)
            
            # ===== Local Constraint Loss =====
            # d(z_t, z_{t+1}) should be close to step_cost
            zy_pred = critic.latent_dynamics(zx, A)
            dist_local = critic.quasimetric_model(zx, zy)
            
            sq_deviation = torch.relu(dist_local - self.local_constraint_step_cost).square().mean()
            violation = sq_deviation - self.local_constraint_epsilon ** 2
            
            # Lagrange multiplier (minimax optimization)
            lag_mult = F.softplus(lagrange_mult)
            local_loss = violation * lag_mult.detach()  # don't backprop through lagrange mult here
            lagrange_loss = -violation.detach() * lag_mult  # minimax: maximize violation w.r.t. multiplier
            
            # ===== Latent Dynamics Loss =====
            # Predicted next latent should match actual next latent
            dist_dynamics = critic.quasimetric_model(zy_pred, zy, bidirectional=True)
            dynamics_loss = dist_dynamics.square().mean() * self.latent_dynamics_weight
            
            # ===== Global Push Loss =====
            # Encourage separation between random pairs
            zy_rolled = torch.roll(zy, 1, dims=0)  # random pairing
            dist_global = critic.quasimetric_model(zx, zy_rolled)
            # Transform to penalize large distances less
            tsfm_dist = F.softplus(
                self.global_push_offset - dist_global, 
                beta=self.global_push_beta
            ).mean()
            global_loss = tsfm_dist * self.global_push_weight
            
            # Total critic loss
            critic_loss = local_loss + dynamics_loss + global_loss
            
            # Update critic
            critic_optim.zero_grad()
            (critic_loss * self.args.loss_scale).backward()
            sync_grads(critic)
            critic_optim.step()
            
            # Update lagrange multiplier
            lagrange_optim.zero_grad()
            (lagrange_loss * self.args.loss_scale).backward()
            lagrange_optim.step()
            
            # Store info
            losses_info[f'critic_{idx}_local'] = local_loss.item()
            losses_info[f'critic_{idx}_dynamics'] = dynamics_loss.item()
            losses_info[f'critic_{idx}_global'] = global_loss.item()
            losses_info[f'critic_{idx}_total'] = critic_loss.item()
            losses_info[f'critic_{idx}_dist_local'] = dist_local.mean().item()
            losses_info[f'critic_{idx}_lagrange_mult'] = lag_mult.item()
        
        return losses_info
    
    def _compute_actor_loss(self, S, G):
        """Compute actor loss using min-distance objective"""
        # Random goal pairing (for exploration)
        G_random = torch.roll(G, 1, dims=0)
        
        # Get action from actor
        A_pred = self.actor(S, G_random)
        
        # Compute distances from all critics
        dists = []
        for critic in self.critics:
            with torch.no_grad():
                SG = torch.cat([S, G_random], dim=-1)
                zs = critic.encoder(SG)
            
            # Predict next latent (with gradients through actor)
            zs_next = critic.latent_dynamics(zs, A_pred)
            
            # Encode goal
            GG = torch.cat([G_random, G_random], dim=-1)
            with torch.no_grad():
                zg = critic.encoder(GG)
            
            # Distance to goal
            dist = critic.quasimetric_model(zs_next, zg)
            dists.append(dist)
        
        # Use pessimistic (max) distance
        max_dist = torch.stack(dists, dim=-1).max(dim=-1).values.mean()
        
        # Actor loss: minimize distance to goal
        actor_loss = max_dist
        actor_loss += self.args.action_l2 * (A_pred / self.args.max_action).pow(2).mean()
        
        return actor_loss, {
            'actor_dist': max_dist.item(),
            'actor_loss': actor_loss.item(),
        }
    
    def _update(self):
        """Single update step"""
        transition = self.buffer.sample(self.args.batch_size)
        
        S = transition['S']
        NS = transition['NS']
        A = transition['A']
        G = transition['G']
        NG = transition['NG']
        
        # Convert to tensors
        A = numpy2torch(A, unsqueeze=False, cuda=self.args.cuda)
        S, G = self._preproc_inputs(S, G)
        NS, NG = self._preproc_inputs(NS, NG)
        
        # Update critics
        critic_losses = self._compute_critic_losses(S, A, G, NS, NG)
        
        # Update actor
        actor_loss, actor_info = self._compute_actor_loss(S, G)
        self.actor_optim.zero_grad()
        (actor_loss * self.args.loss_scale).backward()
        actor_grad_norm = sync_grads(self.actor)
        self.actor_optim.step()
        
        # Soft update target networks
        self._soft_update(self.actor_target, self.actor)
        for critic_target, critic in zip(self.critics_target, self.critics):
            self._soft_update(critic_target, critic)
        
        # Combine all info
        info = {**critic_losses, **actor_info, 'actor_grad_norm': actor_grad_norm}
        return info
    
    def learn(self):
        """Main training loop"""
        if MPI.COMM_WORLD.Get_rank() == 0:
            t0 = time.time()
            stats = {
                'successes': [],
                'hitting_times': [],
                'actor_losses': [],
                'critic_losses': [],
            }
        
        # Prefill buffer
        self.prefill_buffer()
        
        for epoch in range(self.args.n_epochs):
            AL, CL = [], []
            
            for _ in range(self.args.n_cycles):
                # Collect rollout
                (S, A, AG, G), success = self.collect_rollout()
                self.buffer.store_episode(S, A, AG, G)
                self._update_normalizer(S, A, AG, G)
                
                # Update networks
                for _ in range(self.args.n_batches):
                    info = self._update()
                    AL.append(info.get('actor_loss', 0))
                    # Average critic losses
                    cl = np.mean([
                        info.get(f'critic_{i}_total', 0) 
                        for i in range(self.num_critics)
                    ])
                    CL.append(cl)
            
            # Evaluate
            global_success_rate, global_hitting_time = self.eval_agent()
            
            if MPI.COMM_WORLD.Get_rank() == 0:
                t1 = time.time()
                AL = np.array(AL)
                CL = np.array(CL)
                
                stats['successes'].append(global_success_rate)
                stats['hitting_times'].append(global_hitting_time)
                stats['actor_losses'].append(AL.mean())
                stats['critic_losses'].append(CL.mean())
                
                print(f"[info] epoch {epoch:3d} success rate {global_success_rate:6.4f} | " +
                      f"actor loss {AL.mean():6.4f} | critic loss {CL.mean():6.4f} | " +
                      f"time {(t1-t0)/60:6.4f} min")
                
                self.save_model(stats)