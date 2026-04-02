"""
Modified main.py for GCHR Framework with CRL and RIS agent support.
"""

import gym
import numpy as np
import os
import random
import torch
import sys

from mpi4py import MPI

from src.args import get_args
from src.agent import (DDPG, HER, MHER, GCSL, WGCSL, SAC, SAC_HER,
                        TDInfoNCE, QRL, GCHR, PPO_HER, TD3_HER,
                        ContrastiveRL, RIS)

try:
    from env_factory import make_env, setup, update_args_with_env_params
    ENV_FACTORY_AVAILABLE = True
except ImportError:
    ENV_FACTORY_AVAILABLE = False
    print("[Warning] env_factory not found. Using original make_env and setup.")

torch.cuda.current_device()
torch.cuda._initialized = True
print(torch.cuda.is_available())


def make_env_original(args):
    dic = {
        'FetchReach': 'FetchReach-v1',
        'FetchPush': 'FetchPush-v1',
        'FetchSlide': 'FetchSlide-v1',
        'FetchPick': 'FetchPickAndPlace-v1',
        'HandReach': 'HandReach-v0',
        'HandManipulateBlockRotateZ': 'HandManipulateBlockRotateZ-v0',
        'HandManipulateBlockRotateParallel': 'HandManipulateBlockRotateParallel-v0',
        'HandManipulateBlockRotateXYZ': 'HandManipulateBlockRotateXYZ-v0',
        'HandManipulateBlockFull': 'HandManipulateBlockFull-v0',
        'HandManipulateEggRotate': 'HandManipulateEggRotate-v0',
        'HandManipulateEggFull': 'HandManipulateEggFull-v0',
        'HandManipulatePenRotate': 'HandManipulatePenRotate-v0',
        'HandManipulatePenFull': 'HandManipulatePenFull-v0',
    }
    env_id = args.env_name
    try:
        env = gym.make(dic[env_id])
    except:
        raise Exception(f"[error] unknown environment name {args.env_name}")

    default_max_episode_steps = 50
    if hasattr(env, '_max_episode_steps'):
        args.max_episode_steps = env._max_episode_steps
    else:
        args.max_episode_steps = default_max_episode_steps
    print(f"[info] max_episode_steps: {args.max_episode_steps}")
    return env


def setup_original(args, env):
    obs = env.reset()
    o, ag, g = obs['observation'], obs['achieved_goal'], obs['desired_goal']

    args.dim_state = o.shape[0]
    args.dim_goal = g.shape[0]
    args.dim_action = env.action_space.shape[0]
    args.max_action = env.action_space.high[0]

    print(f"[info] args.dim_state: {args.dim_state}")
    print(f"[info] args.dim_goal: {args.dim_goal}")
    print(f"[info] args.dim_action: {args.dim_action}")
    print(f"[info] args.max_action: {args.max_action}")

    start_idx = None
    for i in range(args.dim_state - args.dim_goal + 1):
        sub_o = o[i:i+args.dim_goal]
        if (sub_o == ag).sum() == args.dim_goal:
            start_idx = i
            break

    print(f"[info] start_idx: {start_idx}")
    args.goal_idx = torch.arange(start_idx, start_idx + args.dim_goal)
    print(f"[info] goal_idx: {args.goal_idx}")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    suffix = "(+)rew" if not args.negative_reward else "(-)rew"
    if args.agent in ["her", "mher", "ddpg", "wgcsl", "dsim", "sac", "sac_her",
                       "td_infonce", "gchr", "contrastive_rl", "ris"]:
        suffix += f"_{args.critic}"
        if args.critic != "monolithic":
            suffix += f"_emb{args.dim_embed}"
        if args.terminate:
            suffix += "_terminate"

    args.experiment_name = f"{args.env_name}_{args.agent}_{suffix}_lr{args.lr_critic}_sd{args.seed}"

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"[info] start experiment {args.experiment_name}")


def main(args):
    if ENV_FACTORY_AVAILABLE:
        print("[info] Using unified environment factory (Gym + HIGL support)")
        update_args_with_env_params(args)
        env = make_env(args)
        setup(args, env)
    else:
        print("[info] Using original environment factory (Gym only)")
        env = make_env_original(args)
        setup_original(args, env)

    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    setup(args, env) if ENV_FACTORY_AVAILABLE else setup_original(args, env)

    agent_map = {
        'ddpg'          : DDPG,
        'her'           : HER,
        'mher'          : MHER,
        'gcsl'          : GCSL,
        'wgcsl'         : WGCSL,
        'sac'           : SAC,
        'sac_her'       : SAC_HER,
        'td_infonce'    : TDInfoNCE,
        'qrl'           : QRL,
        'gchr'          : GCHR,
        'ppo_her'       : PPO_HER,
        'td3_her'       : TD3_HER,
        'contrastive_rl': ContrastiveRL,   # <-- NEW
        'ris'           : RIS,             # <-- NEW
    }

    agent = agent_map[args.agent](args, env)
    agent.learn()


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '16'
    os.environ['MKL_NUM_THREADS'] = '16'
    os.environ['IN_MPI'] = '16'
    args = get_args()
    main(args)
