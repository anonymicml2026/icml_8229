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

"""
Actionable Model with HER (MPI-version)

"""
class ActionableModel(Agent):
    def __init__(self, args, env):
        super().__init__(args, env) 
        # create the network
        self.args = args
        self.args.gofar =True
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor
        self.actor = GofarActor(self.args).to(self.device)
        sync_networks(self.actor)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_actor)
        self.actor_target = GofarActor(self.args).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic
        self.critic 		= GofarCritic(self.args).to(self.device)
        sync_networks(self.critic)
        self.critic_target 	= GofarCritic(self.args).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_critic)

        self.buffer = GofarReplayBuffer(args, self.sampler.sample_gofar_transitions)

    def _deterministic_action(self, input_tensor):
        action = self.actor(input_tensor)
        return action
    
    # update the network
    def _update(self):
        transition = self.sample_batch()
        S  = transition['S']
        NS = transition['NS']
        A  = transition['A']
        G  = transition['G']
        R  = transition['R']
        NG = transition['NG']
        # S/NS: (batch, dim_state)
        # A: (batch, dim_action)
        # G: (batch, dim_goal)
        # apply AM goal-chaining (later half of the batch)
        half_batch_size = int(self.args.batch_size / 2)
        her_goals = G[half_batch_size:]
        np.random.shuffle(her_goals)
        G[half_batch_size:] = her_goals

        # start to do the update
        obs_norm = self.s_norm.normalize(S)
        g_norm = self.g_norm.normalize(G)
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=-1)
        obs_next_norm = self.s_norm.normalize(NS)
        g_next_norm = self.g_norm.normalize(NG)
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=-1)
        # transfer them into the tensor
        inputs_norm_tensor = numpy2torch(inputs_norm, unsqueeze=False, cuda=self.args.cuda)
        inputs_next_norm_tensor = numpy2torch(inputs_next_norm, unsqueeze=False, cuda=self.args.cuda)
        actions_tensor = numpy2torch(A, unsqueeze=False, cuda=self.args.cuda)
        r_tensor = numpy2torch(R, unsqueeze=False, cuda=self.args.cuda)

        # Goal-Chaining reward: assign Q(s,a,g) as the reward for randomly sampled goals
        with torch.no_grad():
            r_tensor[half_batch_size:] = self.critic(inputs_norm_tensor, actions_tensor)[half_batch_size:]

        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target(inputs_next_norm_tensor)
            q_next_value = self.critic_target(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()

        # add AM penalty
        num_random_actions = 10
        random_actions_tensor = torch.FloatTensor(q_next_value.shape[0] * num_random_actions, actions_tensor.shape[-1]).uniform_(-1, 1).to(actions_tensor.device)
        inputs_norm_tensor_repeat = inputs_norm_tensor.repeat_interleave(num_random_actions, axis=0)

        q_random_actions = self.critic(inputs_norm_tensor_repeat, random_actions_tensor)
        q_random_actions = q_random_actions.reshape(q_next_value.shape[0], -1)

        # sample according to exp(Q)
        sampled_random_actions = torch.distributions.Categorical(logits=q_random_actions.detach()).sample()
        critic_loss_AM = q_random_actions[torch.arange(q_random_actions.shape[0]), sampled_random_actions].mean()
        critic_loss += critic_loss_AM

        # the actor loss
        actions_real = self.actor(inputs_norm_tensor)
        actor_loss = -self.critic(inputs_norm_tensor, actions_real).mean()
        
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        actor_grad_norm = sync_grads(self.actor)
        self.actor_optim.step()

        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        critic_grad_norm = sync_grads(self.critic)
        self.critic_optim.step()
        return actor_loss.item(), critic_loss.item(), actor_grad_norm, critic_grad_norm

    def learn(self):
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

        # put something to the buffer first
        self.prefill_buffer()

        for epoch in range(self.args.n_epochs):
            AL, CL, AGN, CGN = [], [], [], []

            for _ in range(self.args.n_cycles):
                (S, A, AG, G), success = self.collect_rollout()
                self.buffer.store_episode(S, A, AG, G)
                self._update_normalizer(S, A, AG, G)
                for _ in range(self.args.n_batches):
                    a_loss, c_loss, a_gn, c_gn = self._update()
                    AL.append(a_loss); CL.append(c_loss)
                    AGN.append(a_gn); CGN.append(c_gn)

                self._soft_update(self.actor_target, self.actor)
                self._soft_update(self.critic_target, self.critic)

            global_success_rate, global_hitting_time = self.eval_agent()

            if MPI.COMM_WORLD.Get_rank() == 0:
                t1 = time.time()
                AL = np.array(AL); CL = np.array(CL)
                AGN = np.array(AGN); CGN = np.array(CGN)
                stats['successes'].append(global_success_rate)
                stats['hitting_times'].append(global_hitting_time)
                stats['actor_losses'].append(AL.mean())
                stats['critic_losses'].append(CL.mean())
                stats['actor_grad_norms'].append(AGN.mean())
                stats['critic_grad_norms'].append(CGN.mean())
                print(f"[info] epoch {epoch:3d} success rate {global_success_rate:6.4f} | "+\
                        f" actor loss {AL.mean():6.4f} | critic loss {CL.mean():6.4f} | "+\
                        f" actor gradnorm {AGN.mean():6.4f} | critic gradnorm {CGN.mean():6.4f} | "+\
                        f"time {(t1-t0)/60:6.4f} min")
                self.save_model(stats)

