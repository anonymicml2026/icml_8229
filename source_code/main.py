import gym
import numpy as np
import os
import random
import torch

from mpi4py import MPI
from src.args import get_args
from src.agent import DDPG, HER, MHER, GCSL, WGCSL,DWSL,GCQS,ActionableModel,GoFar,SMORE
from src.goal_utils import TimeLimit, DEFAULT_ENV_PARAMS

torch.cuda.current_device()
torch.cuda._initialized = True
print(torch.cuda.is_available())

def make_env(args):
    dic = {
        'FetchReach': 'FetchReach-v1',
        'FetchPush' : 'FetchPush-v1',
        'FetchSlide': 'FetchSlide-v1',
        'FetchPick' : 'FetchPickAndPlace-v1',
        'HandReach' : 'HandReach-v0',
        'HandManipulateBlockRotateZ'       : 'HandManipulateBlockRotateZ-v0',
        'HandManipulateBlockRotateParallel': 'HandManipulateBlockRotateParallel-v0',
        'HandManipulateBlockRotateXYZ'     : 'HandManipulateBlockRotateXYZ-v0',
        'HandManipulateBlockFull'          : 'HandManipulateBlockFull-v0',
        'HandManipulateEggRotate'          : 'HandManipulateEggRotate-v0',
        'HandManipulateEggFull'            : 'HandManipulateEggFull-v0',
        'HandManipulatePenRotate'          : 'HandManipulatePenRotate-v0',
        'HandManipulatePenFull'            : 'HandManipulatePenFull-v0',
    }

    env_id = args.env_name
    try:
        env = gym.make(dic[env_id])
    except:
        raise Exception(
                f"[error] unknown environment name {args.env_name}")

    # replace environment specific parameters
    if dic[env_id] in DEFAULT_ENV_PARAMS:
        for k, v in DEFAULT_ENV_PARAMS[dic[env_id]].items():
            setattr(args, k, v)
    
    # let argument know max episode length
    default_max_episode_steps = 50

    if hasattr(env, '_max_episode_steps'):
        args.max_episode_steps = env._max_episode_steps
    else:
        args.max_episode_steps = default_max_episode_steps # otherwise use defaulit max episode steps
    print("max_episode_steps:",args.max_episode_steps)
    return env


def setup(args, env):
    obs = env.reset()
    o, ag, g = obs['observation'], obs['achieved_goal'], obs['desired_goal']

    args.dim_state  = o.shape[0]
    args.dim_goal   = g.shape[0]
    args.dim_action = env.action_space.shape[0]
    args.max_action = env.action_space.high[0]
    print("args.dim_state:",args.dim_state)
    print("args.dim_goal:",args.dim_goal)
    print("args.dim_action:",args.dim_action)
    print("args.max_action:",args.max_action)

    # some hack to get the goal from observations
    start_idx = None
    for i in range(args.dim_state - args.dim_goal + 1):
        sub_o = o[i:i+args.dim_goal]
        if (sub_o == ag).sum() == args.dim_goal:
            start_idx = i
            break

    print("start_idx:",start_idx)
    # get goal index to transform state to goal
    args.goal_idx = torch.arange(start_idx, start_idx+args.dim_goal)
    print("goal_idx:",args.goal_idx)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    suffix = "(+)rew" if not args.negative_reward else "(-)rew"
    if args.agent in ["her", "mher", "ddpg", "wgcsl","dsim"]:
        suffix += f"_{args.critic}"
        if args.critic != "monolithic":
            suffix += f"_emb{args.dim_embed}"
        if args.terminate:
            suffix += "_terminate"

    args.experiment_name = f"{args.env_name}_{args.agent}_{suffix}_lr{args.lr_critic}_sd{args.seed}"
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"[info] start experiment {args.experiment_name}")


def main(args):
    # create environment
    env = make_env(args)

    # control seed
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    # update arguments based on environment
    setup(args, env)

    agent_map = {
        'ddpg'    : DDPG,
        'am'    : ActionableModel,
        'her'     : HER,
        'mher'    : MHER,
        'gcsl'    : GCSL,
        'wgcsl'   : WGCSL,
        'dwsl'     : DWSL,
        'gofar'   : GoFar,
        'gcqs' :GCQS,
        'smore' : SMORE,
    }
    
    agent = agent_map[args.agent](args, env)
    agent.learn()

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '16'
    os.environ['MKL_NUM_THREADS'] = '16'
    os.environ['IN_MPI'] = '16'
    args = get_args()
    main(args)
