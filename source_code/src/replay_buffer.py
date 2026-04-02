import numpy as np
import threading
import os

import json

class ReplayBuffer(object):
    """
    The buffer class that stores past trajectories.
    For each value, the buffer shape is (size, max_episode_steps(+1), dim_x)
    """
    def __init__(self, args, sample_func, transition_sample_func=None):
        self.T = args.max_episode_steps
        self.size = args.buffer_size // self.T 

        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        if transition_sample_func:
            self.transition_sample_func = transition_sample_func

        size = self.size
        self.S  = np.zeros([size, self.T+1, args.dim_state ]).astype(np.float32)
        self.A  = np.zeros([size, self.T,   args.dim_action]).astype(np.float32)
        self.AG = np.zeros([size, self.T+1, args.dim_goal  ]).astype(np.float32)
        self.G  = np.zeros([size, self.T,   args.dim_goal  ]).astype(np.float32)
        self.R = np.zeros([size, self.T]).astype(np.float32)
        self.DONE = np.zeros([size, self.T]).astype(bool)
        self.DISC = np.ones([size, self.T]).astype(np.float32)

        self.lock = threading.Lock()

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx

    def store_episode(self, S, A, AG,G):
        n_episodes = S.shape[0]
        with self.lock:
            idx = self._get_storage_idx(inc=n_episodes)
            self.S[idx]  = S
            self.A[idx]  = A
            self.AG[idx] = AG
            self.G[idx]  = G
            self.n_transitions_stored += self.T * n_episodes

    def store_dwsl_episode(self, S, A, AG, G, R, DONE, DISC):
        n_episodes = S.shape[0]
        with self.lock:
            idx = self._get_storage_idx(inc=n_episodes)
            self.S[idx]  = S
            self.A[idx]  = A
            self.AG[idx] = AG
            self.G[idx]  = G
            self.R[idx] = R
            self.DONE[idx] = DONE
            self.DISC[idx] = DISC
            self.n_transitions_stored += self.T * n_episodes
    
    def sample(self, batch_size):
        with self.lock:
            cs  = self.current_size
            S_  = self.S[:cs]
            A_  = self.A[:cs]
            AG_ = self.AG[:cs]
            G_  = self.G[:cs]
        transitions = self.sample_func(S_, A_, AG_, G_, batch_size)
        return transitions
    
    def dwsl_sample(self, batch_size):
        with self.lock:
            cs  = self.current_size
            S_  = self.S[:cs]
            A_  = self.A[:cs]
            AG_ = self.AG[:cs]
            G_  = self.G[:cs]
            R_  = self.R[:cs]
            DONE_  = self.DONE[:cs]
            DISC_  = self.DISC[:cs]
        transitions = self.sample_func(S_, A_, AG_, G_, R_, DONE_, DISC_, batch_size)
        return transitions

    def save(self,path):
            with self.lock:
                states_save_dir = os.path.join(path,'states.npy')
                actions_save_dir = os.path.join(path,'actions.npy')
                achieved_goals_save_dir = os.path.join(path,'achieved_goals.npy')
                desired_goals_save_dir = os.path.join(path,'desired_goals.npy')
                np.save(states_save_dir,self.S)
                np.save(actions_save_dir,self.A)
                np.save(achieved_goals_save_dir,self.AG)
                np.save(desired_goals_save_dir,self.G)

    def load(self,path):
        with self.lock:
            states_load_dir = os.path.join(path,'states.npy')
            actions_load_dir= os.path.join(path,'actions.npy')
            achieved_goals_load_dir = os.path.join(path,'achieved_goals.npy')
            desired_load_dir = os.path.join(path,'desired_goals.npy')
            self.S = np.load(states_load_dir)
            self.A = np.load(actions_load_dir)
            self.AG = np.load(achieved_goals_load_dir)
            self.G = np.load(desired_load_dir)
