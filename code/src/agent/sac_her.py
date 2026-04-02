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
from src.agent.sac import SAC


class SAC_HER(SAC):
    """
    Soft Actor-Critic with Hindsight Experience Replay
    """
    def __init__(self, args, env):
       super().__init__(args, env)
       self.sample_func = self.sampler.sample_her_transitions
       self.buffer = ReplayBuffer(args, self.sample_func)