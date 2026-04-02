"""
RIS (Reinforcement learning with Imagined Subgoals) Agent
Based on: "Goal-Conditioned RL with Imagined Subgoals"
          (Chane-Sane et al., ICML 2021)

Adapted to the GCHR framework.

Key components:
    1. High-level policy (LaplacePolicy) predicts midpoint subgoals.
    2. Prior policy = actor_target conditioned on the subgoal.
    3. Actor is KL-regularised towards the prior.
    4. Critic is a standard TD-learning ensemble Q.
"""

import copy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from mpi4py import MPI
from src.gcrl_model import GaussianActor, EnsembleCritic, LaplacePolicy
from src.replay_buffer import ReplayBuffer
from src.goal_utils import numpy2torch, sync_networks, sync_grads
from src.agent.base import Agent


class RIS(Agent):
    """
    RIS agent within the GCHR framework.

    During training the high-level policy proposes subgoals halfway between
    state and goal (measured by |V|).  The subgoal-conditioned actor_target
    defines a prior; the online actor is KL-regularised towards this prior.
    At test time only the flat actor is used (no subgoals needed).
    """

    def __init__(self, args, env):
        super().__init__(args, env)

        self.args = args
        self.args.gchr = True           # use GaussianActor path in base
        self.device = torch.device("cuda" if args.cuda else "cpu")

        # hyper-parameters
        self.alpha_ris    = getattr(args, 'ris_alpha', 0.1)
        self.Lambda       = getattr(args, 'ris_lambda', 0.1)
        self.n_subgoals   = getattr(args, 'ris_n_subgoals', 10)
        self.tau          = getattr(args, 'tau', 0.005)
        self.epsilon      = 1e-16

        # Actor
        self.actor = GaussianActor(args).to(self.device)
        sync_networks(self.actor)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(),
                                            lr=args.lr_actor)
        self.actor_target = GaussianActor(args).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic (ensemble Q)
        self.critic = EnsembleCritic(args).to(self.device)
        sync_networks(self.critic)
        self.critic_target = EnsembleCritic(args).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = torch.optim.Adam(self.critic.parameters(),
                                             lr=args.lr_critic)

        # High-level policy (subgoal predictor)
        self.subgoal_net = LaplacePolicy(args).to(self.device)
        sync_networks(self.subgoal_net)
        self.subgoal_optim = torch.optim.Adam(self.subgoal_net.parameters(),
                                              lr=getattr(args, 'h_lr', 1e-4))

        # Sampler
        self.sample_func = self.sampler.sample_ris_transitions
        self.buffer = ReplayBuffer(args, self.sample_func)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _value(self, state, goal):
        """V(s, g) = min_i Q_i(s, pi(s,g), g).
        Inputs must already be normalised tensors."""
        with torch.no_grad():
            action, _, _ = self.actor.sample(state, goal)
        Q = self.critic(state, action, goal)          # (B, n_Q)
        return torch.min(Q, dim=-1, keepdim=True)[0]  # (B, 1)

    def _sample_subgoals(self, state, goal):
        """Sample n_subgoals from the high-level policy.
        Inputs must already be normalised tensors."""
        dist = self.subgoal_net(state, goal)
        sg = dist.rsample((self.n_subgoals,))  # (n_sg, B, dim_goal)
        sg = sg.permute(1, 0, 2)               # (B, n_sg, dim_goal)
        return sg

    def _goal_to_normalised_state(self, goal_norm):
        """Create a normalised fake state from a normalised goal tensor.

        The goal occupies specific indices in the state vector (args.goal_idx).
        We place the normalised goal values at those indices and set the
        remaining dimensions to zero (which, after normalisation, corresponds
        to the current running mean).  This is an approximation used by the
        original RIS paper for computing V(s_g, g).
        """
        B = goal_norm.shape[0]
        fake = torch.zeros(B, self.args.dim_state, device=self.device)
        goal_idx = self.args.goal_idx.long()
        fake[:, goal_idx] = goal_norm
        return fake

    # ------------------------------------------------------------------
    # High-level policy update  (weighted MLE, Eq.7 of paper)
    # ------------------------------------------------------------------
    def _update_subgoal_net(self, S_norm, G_norm, SG_norm):
        """All inputs are already normalised tensors."""
        sg_dist = self.subgoal_net(S_norm, G_norm)

        with torch.no_grad():
            sg_mean = sg_dist.loc                               # (B, dim_g)

            # cost through current subgoal mean
            v1_pol = self._value(S_norm, sg_mean).squeeze(-1)
            v2_pol = self._value(
                self._goal_to_normalised_state(sg_mean),
                G_norm).squeeze(-1)
            cost_pol = torch.max(v1_pol.abs(), v2_pol.abs())

            # cost through candidate subgoal
            v1_cand = self._value(S_norm, SG_norm).squeeze(-1)
            v2_cand = self._value(
                self._goal_to_normalised_state(SG_norm),
                G_norm).squeeze(-1)
            cost_cand = torch.max(v1_cand.abs(), v2_cand.abs())

            adv = -(cost_cand - cost_pol)
            weight = F.softmax(adv / self.Lambda, dim=0)

        log_prob = sg_dist.log_prob(SG_norm).sum(-1)
        loss = -(log_prob * weight).mean()

        self.subgoal_optim.zero_grad()
        loss.backward()
        sync_grads(self.subgoal_net)
        self.subgoal_optim.step()

    # ------------------------------------------------------------------
    # Actor loss with KL-regularisation (Eq.9)
    # ------------------------------------------------------------------
    def _actor_loss_and_kl(self, S_norm, G_norm):
        batch_size = S_norm.shape[0]

        # current policy distribution (pre-tanh)
        action_dist = self.actor(S_norm, G_norm)
        action = action_dist.rsample()                          # (B, dim_a)

        # subgoals from high-level policy
        with torch.no_grad():
            subgoals = self._sample_subgoals(S_norm, G_norm)    # (B, n_sg, dim_g)

        # prior = average subgoal-conditioned target policy
        S_exp = S_norm.unsqueeze(1).expand(
            batch_size, self.n_subgoals, self.args.dim_state)
        prior_dist = self.actor_target(S_exp, subgoals)
        a_exp = action.unsqueeze(1).expand(
            batch_size, self.n_subgoals, self.args.dim_action)
        prior_lp = prior_dist.log_prob(a_exp).sum(-1, keepdim=True).exp()
        prior_log_prob = torch.log(prior_lp.mean(1) + self.epsilon)

        D_KL = action_dist.log_prob(action).sum(-1, keepdim=True) - prior_log_prob

        action_tanh = torch.tanh(action)
        return action_tanh, D_KL

    # ------------------------------------------------------------------
    # Full update step
    # ------------------------------------------------------------------
    def _update(self):
        transition = self.buffer.sample(self.args.batch_size)

        S  = transition['S']
        NS = transition['NS']
        A  = transition['A']
        G  = transition['G']
        R  = transition['R']
        SG = transition['SG']            # subgoal candidate (numpy)

        A = numpy2torch(A, cuda=self.args.cuda)
        R = numpy2torch(R, cuda=self.args.cuda)

        # normalise state, goal, next-state (returns tensors on device)
        S_t,  G_t = self._preproc_inputs(S, G)
        NS_t, _   = self._preproc_inputs(NS)

        # normalise subgoal candidates the same way as goals
        SG_clipped = np.clip(SG, -self.args.clip_obs, self.args.clip_obs)
        SG_t = numpy2torch(self.g_norm.normalize(SG_clipped),
                           cuda=self.args.cuda)

        # ==== 1. Critic (standard TD) ========================================
        with torch.no_grad():
            NA, _, _ = self.actor_target.sample(NS_t, G_t)
            NQ = self.critic_target(NS_t, NA, G_t)
            NQ = torch.min(NQ, dim=-1, keepdim=True)[0]
            clip_return = 1.0 / (1.0 - self.args.gamma)
            if self.args.negative_reward:
                target = (R + self.args.gamma * NQ).clamp(-clip_return, 0)
            else:
                target = (R + self.args.gamma * NQ).clamp(0, clip_return)

        Q = self.critic(S_t, A, G_t)
        critic_loss = 0.5 * (Q - target).pow(2).sum(-1).mean()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic)
        self.critic_optim.step()

        # ==== 2. High-level policy ============================================
        self._update_subgoal_net(S_t, G_t, SG_t)

        # ==== 3. Actor (KL-regularised) =======================================
        A_new, D_KL = self._actor_loss_and_kl(S_t, G_t)
        Q_new = self.critic(S_t, A_new, G_t)
        Q_new = torch.min(Q_new, dim=-1, keepdim=True)[0]
        actor_loss = (self.alpha_ris * D_KL - Q_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor)
        self.actor_optim.step()

        # NOTE: target networks are updated in learn(), not here,
        #       to match the GCHR framework convention.

        return actor_loss.item(), critic_loss.item()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def learn(self):
        if MPI.COMM_WORLD.Get_rank() == 0:
            t0 = time.time()
            stats = {
                'successes': [],
                'hitting_times': [],
                'actor_losses': [],
                'critic_losses': [],
            }

        self.prefill_buffer()

        if self.args.cuda:
            n_scales = (self.args.max_episode_steps
                        * self.args.rollout_n_episodes
                        // (self.args.n_batches * 2)) + 1
        else:
            n_scales = 1

        for epoch in range(self.args.n_epochs):
            AL, CL = [], []
            for _ in range(self.args.n_cycles):
                (S, A, AG, G), success = self.collect_rollout()
                self.buffer.store_episode(S, A, AG, G)
                self._update_normalizer(S, A, AG, G)

                for _ in range(n_scales):
                    for _ in range(self.args.n_batches):
                        a_loss, c_loss = self._update()
                        AL.append(a_loss)
                        CL.append(c_loss)

                    # Soft target updates (once per n_batches block)
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
                print(f"[RIS] epoch {epoch:3d} success {global_success_rate:6.4f} | "
                      f"actor {AL.mean():6.4f} | critic {CL.mean():6.4f} | "
                      f"time {(t1 - t0) / 60:.2f} min")
                self.save_model(stats)
