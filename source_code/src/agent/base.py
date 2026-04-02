import copy
import numpy as np
import time
import torch

from src.gcrl_model import *
from src.replay_buffer import ReplayBuffer
from src.goal_utils import *
from src.sampler import Sampler


################################################################################
#
# Agent class for sparse reward RL
#
################################################################################


class Agent(object):
    """
    The sparse reward RL agent superclass
    """
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.env.action_range = self.to_tensor(self.env.action_space.high)
        self.env.action_size = int(args.max_action)
        self.best_success_rate = 0.0

        self.args.gcqs = False
        self.actor = Actor(args)
        sync_networks(self.actor)

        if self.args.cuda:
            self.actor.cuda()

        self.actor_optim  = torch.optim.Adam(self.actor.parameters(),
                                             lr=self.args.lr_actor)

        # observation/goal normalizer
        self.s_norm = Normalizer(size=args.dim_state,
                                 default_clip_range=self.args.clip_range)
        self.g_norm = Normalizer(size=args.dim_goal,
                                 default_clip_range=self.args.clip_range)
        self.sampler = Sampler(args, self.env.compute_reward, self.env)
        self.sample_func = self.sampler.sample_ddpg_transitions

    def to_tensor(self,x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32))
        else:
            return torch.FloatTensor(x)

    def _preproc_inputs(self, s=None, g=None, unsqueeze=False):
        s_tensor = None
        if s is not None:
            s_ = self.s_norm.normalize(
                    np.clip(s, -self.args.clip_obs, self.args.clip_obs))
            s_tensor = numpy2torch(s_, unsqueeze=unsqueeze, cuda=self.args.cuda)

        g_tensor = None
        if g is not None:
            g_ = self.g_norm.normalize(
                    np.clip(g, -self.args.clip_obs, self.args.clip_obs))
            g_tensor = numpy2torch(g_, unsqueeze=unsqueeze, cuda=self.args.cuda)
        return s_tensor, g_tensor

    def _select_actions(self, s_tensor, g_tensor, stochastic):
        with torch.no_grad():
            a = self.actor(s_tensor, g_tensor).detach().cpu().squeeze().numpy()

        if not stochastic:
            return a

        # gaussian noise
        max_action = self.args.max_action
        a += self.args.noise_eps * max_action * np.random.randn(*a.shape)
        a = np.clip(a, -max_action, max_action)

        # eps-greedy
        rand_a = np.random.uniform(low=-max_action, high=max_action,
                                   size=self.args.dim_action)

        # choose if use the random actions
        a += np.random.binomial(1, self.args.random_eps, 1)[0] * (rand_a - a)
        return a
    
    def _select_gofar_actions(self, s_tensor, g_tensor, stochastic):
        with torch.no_grad():
            input_tensor = torch.cat([s_tensor, g_tensor], -1)
            a = self.actor(input_tensor).detach().cpu().squeeze().numpy()

        if not stochastic:
            return a

        # gaussian noise
        max_action = self.args.max_action
        a += self.args.noise_eps * max_action * np.random.randn(*a.shape)
        a = np.clip(a, -max_action, max_action)

        # eps-greedy
        rand_a = np.random.uniform(low=-max_action, high=max_action,
                                   size=self.args.dim_action)

        # choose if use the random actions
        a += np.random.binomial(1, self.args.random_eps, 1)[0] * (rand_a - a)
        return a
    
    def _select_gcqs_actions(self, s_tensor, g_tensor, stochastic):
        with torch.no_grad():
            a,_,_ = self.actor.sample(s_tensor, g_tensor)
            a = a.cpu().data.numpy().squeeze()
        if not stochastic:
            return a

        # gaussian noise
        max_action = self.args.max_action
        a += self.args.noise_eps * max_action * np.random.randn(*a.shape)
        a = np.clip(a, -max_action, max_action)

        # eps-greedy
        rand_a = np.random.uniform(low=-max_action, high=max_action,
                                   size=self.args.dim_action)

        # choose if use the random actions
        a += np.random.binomial(1, self.args.random_eps, 1)[0] * (rand_a - a)
        return a
    
    def _update_normalizer(self, S, A, AG,G):
        transition = self.sample_func(
                S, A, AG, G, A.shape[1])

        S_ = transition['S']
        G_ = transition['G']

        S_ = np.clip(S_, -self.args.clip_obs, self.args.clip_obs)
        G_ = np.clip(G_, -self.args.clip_obs, self.args.clip_obs)
        
        self.s_norm.update(S_)
        self.g_norm.update(G_)

        self.s_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def dwsl_update_normalizer(self, S, A, AG,G, R, DONE, DISC):
        transition = self.sample_func(
                S, A, AG, G, R, DONE, DISC, A.shape[1])

        S_ = transition['S']
        G_ = transition['G']

        S_ = np.clip(S_, -self.args.clip_obs, self.args.clip_obs)
        G_ = np.clip(G_, -self.args.clip_obs, self.args.clip_obs)
        
        self.s_norm.update(S_)
        self.g_norm.update(G_)

        self.s_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            d = self.args.polyak
            tp.data.copy_( (1-d) * sp.data + d * tp.data )

    def _update(self):
        pass

    def select_action(self, o, stochastic=True):
        s  = o['observation'].astype(np.float32)
        g  = o['desired_goal'].astype(np.float32)
        s_tensor, g_tensor = self._preproc_inputs(s, g, unsqueeze=True)
        if self.args.gcqs:
            actions = self._select_gcqs_actions(s_tensor, g_tensor, stochastic)
        elif self.args.gofar:
            actions = self._select_gofar_actions(s_tensor, g_tensor, stochastic)
        else:
            actions = self._select_actions(s_tensor, g_tensor, stochastic)
        return actions

    def prefill_buffer(self):
        if self.args.n_init_episodes == 0:
            return
        n_threads = MPI.COMM_WORLD.Get_size()
        n_collects = (self.args.n_init_episodes // self.args.rollout_n_episodes // n_threads+1)
        for _ in range(n_collects):
            (S, A, AG, G), _ = self.collect_rollout(uniform_random_action=True)
            self.buffer.store_episode(S, A, AG, G)

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    def sample_batch(self):
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g, = transitions['S'], transitions['NS'], transitions['G']
        #transitions['IS'] = np.clip(transitions['IS'], -self.args.clip_obs, self.args.clip_obs)
        transitions['S'], transitions['G'] = self._preproc_og(o, g)
        transitions['NS'], transitions['NG'] = self._preproc_og(o_next, g)

        self.s_norm.update(transitions['S'])
        self.g_norm.update(transitions['G'])
        self.s_norm.recompute_stats()
        self.g_norm.recompute_stats()

        return transitions

    def collect_rollout(self, uniform_random_action=False, stochastic=True):
        n_episodes = self.args.rollout_n_episodes
        dim_state  = self.args.dim_state
        dim_action = self.args.dim_action
        dim_goal   = self.args.dim_goal
        T = self.args.max_episode_steps
        max_action = self.args.max_action

        S  = np.zeros((n_episodes, T+1, dim_state),  np.float32)
        A  = np.zeros((n_episodes, T,   dim_action), np.float32)
        AG = np.zeros((n_episodes, T+1, dim_goal),   np.float32)
        G  = np.zeros((n_episodes, T,   dim_goal),   np.float32)
        success = np.zeros((n_episodes), np.float32)

        for i in range(n_episodes):
            o = self.env.reset()
            for t in range(T):
                if uniform_random_action:
                    a = np.random.uniform(low=-max_action,
                                          high=max_action,
                                          size=(dim_action,))
                else:
                    a = self.select_action(o, stochastic=stochastic)
                new_o, r, d, info = self.env.step(a)
                S[i][t]  = o['observation'].copy()
                AG[i][t] = o['achieved_goal'].copy()
                G[i][t]  = o['desired_goal'].copy()
                A[i][t]  = a.copy()
                o = new_o

            success[i] = info['is_success']
            S[i][t+1] = o['observation'].copy()
            AG[i][t+1] = o['achieved_goal'].copy()
        return (S, A, AG,G), success
    
    def dwsl_collect_rollout(self, uniform_random_action=False, stochastic=True):
        n_episodes = self.args.rollout_n_episodes
        dim_state  = self.args.dim_state
        dim_action = self.args.dim_action
        dim_goal   = self.args.dim_goal
        T = self.args.max_episode_steps
        max_action = self.args.max_action

        S  = np.zeros((n_episodes, T+1, dim_state),  np.float32)
        A  = np.zeros((n_episodes, T,   dim_action), np.float32)
        AG = np.zeros((n_episodes, T+1, dim_goal),   np.float32)
        G  = np.zeros((n_episodes, T,   dim_goal),   np.float32)
        R = np.zeros((n_episodes, T),   np.float32)
        DONE = np.zeros((n_episodes, T),   bool)
        DISC = np.ones((n_episodes, T),   np.float32)
        success = np.zeros((n_episodes), np.float32)

        for i in range(n_episodes):
            o = self.env.reset()
            for t in range(T):
                if uniform_random_action:
                    a = np.random.uniform(low=-max_action,
                                          high=max_action,
                                          size=(dim_action,))
                else:
                    a = self.select_action(o, stochastic=stochastic)
                new_o, r, _, info = self.env.step(a)
                S[i][t]  = o['observation'].copy()
                AG[i][t] = o['achieved_goal'].copy()
                G[i][t]  = o['desired_goal'].copy()
                A[i][t]  = a.copy()
                R[i][t] = r.copy()
                if self.env.distance_threshold is not None:
                    done = (np.linalg.norm(o['achieved_goal'] - o['desired_goal']) < self.env.distance_threshold)
                if "discount" in info:
                   DISC[i][t] = info["discount"]
                else:
                   DISC[i][t]  = 1 - float(done)
                DONE[i][t] = done
                o = new_o
            success[i] = info['is_success']
            S[i][t+1] = o['observation'].copy()
            AG[i][t+1] = o['achieved_goal'].copy()
        return (S, A, AG,G,R,DONE,DISC), success

    def eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.eval_rollout_n_episodes):
            per_success_rate = []
            o = self.env.reset()
            for _ in range(self.args.max_episode_steps):
                with torch.no_grad():
                    a = self.select_action(o, stochastic=False)
                o, _, _, info = self.env.step(a)
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        local_hitting_time = first_nonzero(total_success_rate, 1, self.args.max_episode_steps+1).mean()
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        global_hitting_time = MPI.COMM_WORLD.allreduce(local_hitting_time, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size(), global_hitting_time / MPI.COMM_WORLD.Get_size()

    def save_model(self, stats):
        sd = {
            "actor": self.actor.state_dict(),
            "args": self.args,
            "stats": stats,
            "s_mean": self.s_norm.mean, 
            "s_std": self.s_norm.std,
            "g_mean": self.g_norm.mean,
            "g_std": self.g_norm.std,
        }
        if hasattr(self, "critic"):
            sd["critic"] = self.critic.state_dict()

        torch.save(sd,
            os.path.join(self.args.save_dir, f"{self.args.experiment_name}.pt"))
        if stats['successes'][-1] > self.best_success_rate:
            self.best_success_rate = stats['successes'][-1]
            torch.save(sd,
                os.path.join(self.args.save_dir, f"{self.args.experiment_name}_best.pt"))
            
    def learn(self):
        pass
