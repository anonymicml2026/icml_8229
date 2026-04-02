import copy
import numpy as np
import time
import torch
import itertools
from typing import Dict, Optional

from src.gcrl_model import *
from src.replay_buffer import ReplayBuffer
from src.goal_utils import *
from src.agent.base import Agent

class DWSL(Agent):
    """
    Distance weighted supervised learning agent
    """
    def __init__(self, args, env):
        super().__init__(args, env)
        self.args = args
        self.beta = self.args.dwsl_beta
        self.alpha = self.args.dwsl_alpha
        self.clip_score = self.args.clip_score
        self.nstep = self.args.nstep
        self.bins = self.args.bins
        self.action_range = [
            -self.args.max_action,
            self.args.max_action,
        ]

        self.critic = DiscreteMLPDistance(args,self.bins)
        sync_networks(self.critic)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        if self.args.cuda:
            self.critic.cuda()
            self.critic_target.cuda()
        self.critic_optim  = torch.optim.Adam(self.critic.parameters(),
                                              lr=self.args.lr_critic)

        def sample_func(S, A, AG, G, R, DONE, DISC, size):
            return self.sampler.sample_dwsl_transitions(
                    S, A, AG, G, R, DONE, DISC, size)

        self.sample_func = sample_func
        self.buffer = ReplayBuffer(args, self.sample_func)

    def _get_distance(self, logits: torch.Tensor) -> torch.Tensor:
        distribution = torch.nn.functional.softmax(logits, dim=-1)  # (E, B, D)
        distances = torch.arange(start=0, end=self.bins, device=logits.device) / self.bins
        distances = distances.unsqueeze(0).unsqueeze(0)  # (E, B, D)
        if self.alpha is None:
            # Return the expectation
            predicted_distance = (distribution * distances).sum(dim=-1)
        else:
            # Return the LSE weighted by the distribution.
            exp_q = torch.exp(-distances / self.alpha)
            predicted_distance = -self.alpha * torch.log(torch.sum(distribution * exp_q, dim=-1))
        return torch.max(predicted_distance, dim=0)[0]

    def _update(self):
        transition = self.buffer.dwsl_sample(self.args.batch_size)
        S  = transition['S']
        NS = transition['NS']
        A  = transition['A']
        G  = transition['G']
        R  = transition['R']
        H = transition['H']
        DISC = transition['DISC']
        # S: (batch, dim_state)
        # A: (batch, dim_action)
        # G: (batch, dim_goal)
        # W: (batch, 1)
        R = numpy2torch(R, unsqueeze=False, cuda=self.args.cuda)
        A = numpy2torch(A, unsqueeze=False, cuda=self.args.cuda)
        H = numpy2torch(H, unsqueeze=False, cuda=self.args.cuda)
        DISC = numpy2torch(DISC, unsqueeze=False, cuda=self.args.cuda)
        S, G = self._preproc_inputs(S, G)
        NS, _ = self._preproc_inputs(NS)

        with torch.no_grad():
            empirical_targets = (H - 1) // self.nstep
            empirical_targets[empirical_targets < 0] = self.bins - 1  # Set to max bin value (= horizon by default)
            empirical_targets[empirical_targets >= self.bins] = self.bins - 1
            # Now one-hot the empirical distribution
            target_distribution = torch.nn.functional.one_hot(empirical_targets, num_classes=self.bins)
            target_distribution = target_distribution.unsqueeze(0)  # (1, B, D)

        # Now train the distance function with NLL loss
        logits = self.critic(S,G)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        critic_loss = (-target_distribution * log_probs).sum(dim=-1).mean()  # Sum over D dim, avg over B and E

        # Compute the Actor Loss
        with torch.no_grad():
            # Compute the advantage. This is equal to Q(s,a) - V(s) normally
            # But, we are using costs. Thus the advantage is V(s) - Q(s,a) = V(s) - c(s,a) - V(s')
            # First compute the cost tensor. This is zero unless the horizon is in nstep
            cost = torch.logical_or(H >= self.nstep, H < 0).float() / self.bins
            distance = self._get_distance(logits)
            next_distance = self._get_distance(self.critic(NS,  G))
            adv = distance - cost - DISC * next_distance
            exp_adv = torch.exp(adv / self.beta)
            if self.clip_score is not None:
                exp_adv = torch.clamp(exp_adv, max=self.clip_score)

        dist = self.actor(S,G)
        if isinstance(dist, torch.distributions.Distribution):
            bc_loss = -dist.log_prob(A).sum(dim=-1)
        elif torch.is_tensor(dist):
            assert dist.shape ==A.shape
            bc_loss = torch.nn.functional.mse_loss(dist, A, reduction="none").sum(dim=-1)
        else:
            raise ValueError("Invalid policy output provided")
        assert exp_adv.shape == bc_loss.shape
        actor_loss = (exp_adv * bc_loss).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic)
        self.critic_optim.step()
        return actor_loss.item(), critic_loss.item(), distance.mean().item(), adv.mean().item()

    def learn(self):
        if MPI.COMM_WORLD.Get_rank() == 0:
            t0 = time.time()
            successes = []
            hitting_times = []
            stats = {
                'successes': [],
                'hitting_times': [],
                'actor_losses': [],
                'critic_losses': [],
                'distance':[],
                'advantage':[],
            }

        # put something to the buffer first
        self.prefill_buffer()
        if self.args.cuda:
            n_scales = (self.args.max_episode_steps * self.args.rollout_n_episodes // (self.args.n_batches*2)) + 1
        else:
            n_scales = 1

        for epoch in range(self.args.n_epochs):
            AL, CL ,DIS , ADV = [], [] , [] , []
            for _ in range(self.args.n_cycles):
                (S, A, AG, G, R, DONE, DISC), success = self.dwsl_collect_rollout()
                self.buffer.store_dwsl_episode(S, A, AG, G, R, DONE, DISC)
                self.dwsl_update_normalizer(S, A, AG, G, R, DONE, DISC)

                for _ in range(n_scales): # scale up for single thread
                    for _ in range(self.args.n_batches):
                        a_loss, c_loss , distance , adv = self._update()
                        AL.append(a_loss); CL.append(c_loss); DIS.append(distance); ADV.append(adv)

                    self._soft_update(self.actor_target, self.actor)
                    self._soft_update(self.critic_target, self.critic)

            global_success_rate, global_hitting_time = self.eval_agent()

            if MPI.COMM_WORLD.Get_rank() == 0:
                t1 = time.time()
                AL = np.array(AL); CL = np.array(CL); DIS = np.array(DIS); ADV = np.array(ADV)
                stats['successes'].append(global_success_rate)
                stats['hitting_times'].append(global_hitting_time)
                stats['actor_losses'].append(AL.mean())
                stats['critic_losses'].append(CL.mean())
                stats['distance'].append(DIS.mean())
                stats['advantage'].append(ADV.mean())
                print(f"[info] epoch {epoch:3d} success rate {global_success_rate:6.4f} | "+\
                        f"time {(t1-t0)/60:6.4f} min")
                self.save_model(stats)
