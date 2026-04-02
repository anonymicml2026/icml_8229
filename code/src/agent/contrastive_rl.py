"""
Contrastive RL (NCE) Agent
Based on: "Contrastive Learning as Goal-Conditioned Reinforcement Learning"
          (Eysenbach et al., NeurIPS 2022)

Adapted to the GCHR framework.
"""

import copy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from mpi4py import MPI
from src.gcrl_model import GaussianActor, ContrastiveCritic
from src.replay_buffer import ReplayBuffer
from src.goal_utils import numpy2torch, sync_networks, sync_grads
from src.agent.base import Agent


class ContrastiveRL(Agent):
    """
    Contrastive RL (NCE) agent.

    Critic: inner product between phi(s,a) and psi(g).
    Critic loss: BCE on the (B x B) logits matrix (diagonal = positive).
    Actor loss: maximise diagonal of the logits matrix.
    """

    def __init__(self, args, env):
        super().__init__(args, env)

        self.args = args
        self.args.gchr = True                   # use GaussianActor path in base
        self.device = torch.device("cuda" if args.cuda else "cpu")

        # hyper-parameters
        self.tau = getattr(args, 'tau', 0.005)
        self.random_goals_ratio = getattr(args, 'random_goals', 0.5)
        self.entropy_coef = getattr(args, 'crl_entropy_coef', 0.0)

        # Actor
        self.actor = GaussianActor(args).to(self.device)
        sync_networks(self.actor)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(),
                                            lr=args.lr_actor)

        # Contrastive Critic (NCE does not require a target network)
        self.critic = ContrastiveCritic(args).to(self.device)
        sync_networks(self.critic)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(),
                                             lr=args.lr_critic)

        # Sampler
        self.sample_func = self.sampler.sample_crl_transitions
        self.buffer = ReplayBuffer(args, self.sample_func)

    # ------------------------------------------------------------------
    def _update(self):
        transition = self.buffer.sample(self.args.batch_size)

        S = transition['S']
        A = transition['A']
        G = transition['G']

        A = numpy2torch(A, cuda=self.args.cuda)
        S_t, G_t = self._preproc_inputs(S, G)

        batch_size = S_t.shape[0]

        # ==== 1. Critic loss (NCE-binary) ====================================
        sa_repr, g_repr = self.critic(S_t, A, G_t)
        logits = torch.einsum('ik,jk->ij', sa_repr, g_repr)  # (B, B)

        labels = torch.eye(batch_size, device=self.device)
        critic_loss = F.binary_cross_entropy_with_logits(
            logits, labels, reduction='mean')

        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic)
        self.critic_optim.step()

        # ==== 2. Actor loss ===================================================
        state_for_actor = S_t
        goal_for_actor = G_t

        if self.random_goals_ratio == 0.5:
            state_for_actor = torch.cat([S_t, S_t], dim=0)
            goal_for_actor = torch.cat([G_t, torch.roll(G_t, 1, 0)], dim=0)
        elif self.random_goals_ratio == 1.0:
            goal_for_actor = torch.roll(G_t, 1, 0)

        new_action, log_prob, _ = self.actor.sample(state_for_actor,
                                                     goal_for_actor)
        sa_repr_a, g_repr_a = self.critic(state_for_actor, new_action,
                                           goal_for_actor)
        q_diag = torch.sum(sa_repr_a * g_repr_a, dim=-1, keepdim=True)
        actor_loss = (self.entropy_coef * log_prob - q_diag).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor)
        self.actor_optim.step()

        return actor_loss.item(), critic_loss.item()

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

            global_success_rate, global_hitting_time = self.eval_agent()

            if MPI.COMM_WORLD.Get_rank() == 0:
                t1 = time.time()
                AL = np.array(AL); CL = np.array(CL)
                stats['successes'].append(global_success_rate)
                stats['hitting_times'].append(global_hitting_time)
                stats['actor_losses'].append(AL.mean())
                stats['critic_losses'].append(CL.mean())
                print(f"[CRL] epoch {epoch:3d} success {global_success_rate:6.4f} | "
                      f"actor {AL.mean():6.4f} | critic {CL.mean():6.4f} | "
                      f"time {(t1 - t0) / 60:.2f} min")
                self.save_model(stats)
