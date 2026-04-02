"""
Environment creation functions for GCHR framework with HIGL environments.
This module provides a unified interface to create both Gym (Fetch/Hand) 
and HIGL (PointMaze/AntMaze/Reacher/Pusher) environments.
"""

import gym
import numpy as np
import sys
import os

# Try to import HIGL environments
try:
    # Add HIGL goal_env to path if available
    # Adjust this path based on where you place the HIGL code
    import goal_env
    from goal_env.mujoco import EnvWithGoal
    HIGL_AVAILABLE = True
except ImportError:
    HIGL_AVAILABLE = False
    print("[Warning] HIGL environments not available. Only Gym environments will work.")

from higl_env_wrapper import HIGLtoGCHRWrapper, EnvWithGoalWrapper


def create_higl_maze_env(env_name, reward_shaping='sparse', maze_size_scaling=4, 
                          random_start=True, evaluate=False):
    """
    Create HIGL maze environment (PointMaze or AntMaze).
    
    Args:
        env_name: Environment name (e.g., 'PointMaze-v1', 'AntMaze-v1')
        reward_shaping: 'sparse' or 'dense'
        maze_size_scaling: Scaling factor for maze size (4 for v1, 8 for v0)
        random_start: Whether to start from random positions
        evaluate: Whether in evaluation mode
    
    Returns:
        Wrapped environment compatible with GCHR framework
    """
    if not HIGL_AVAILABLE:
        raise ImportError("HIGL environments not available. Please install goal_env package.")
    
    # Create base HIGL environment
    try:
        # Try using gym.make if environment is registered
        base_env = gym.make(env_name)
    except:
        # If not registered, create manually
        from goal_env.mujoco.create_maze_env import create_maze_env
        
        # Map env_name to HIGL format
        if 'Point' in env_name:
            higl_env_name = 'Point' + env_name.split('Point')[1].split('-')[0]
        elif 'Ant' in env_name:
            higl_env_name = 'Ant' + env_name.split('Ant')[1].split('-')[0]
        else:
            higl_env_name = env_name
        
        base_env = create_maze_env(
            env_name=higl_env_name,
            maze_size_scaling=maze_size_scaling,
            random_start=random_start,
        )
    
    # Wrap with EnvWithGoal for goal management
    step_style = (reward_shaping == 'sparse')
    wrapped_env = EnvWithGoal(
        base_env=base_env,
        env_name=env_name,
        step_style=step_style
    )
    
    # Further wrap for GCHR compatibility
    gchr_env = EnvWithGoalWrapper(
        base_env=wrapped_env,
        env_name=env_name,
        evaluate=evaluate,
        step_style=step_style,
    )
    
    return gchr_env


def create_higl_manipulator_env(env_name, reward_shaping='dense', evaluate=False):
    """
    Create HIGL manipulator environment (Reacher or Pusher).
    
    Args:
        env_name: Environment name (e.g., 'Reacher3D-v0', 'Pusher-v0')
        reward_shaping: 'sparse' or 'dense'
        evaluate: Whether in evaluation mode
    
    Returns:
        Wrapped environment compatible with GCHR framework
    """
    if not HIGL_AVAILABLE:
        raise ImportError("HIGL environments not available. Please install goal_env package.")
    
    try:
        # Try using gym.make if environment is registered
        base_env = gym.make(env_name)
    except:
        # If not registered, create manually
        from goal_env.mujoco.create_fetch_env import create_fetch_env
        base_env = create_fetch_env(
            env_name=env_name,
            reward_shaping=reward_shaping,
        )
    
    # The create_fetch_env already returns a wrapped environment
    # We just need to ensure it's compatible with GCHR
    if not isinstance(base_env.observation_space, gym.spaces.Dict):
        goal_dim = 3 if 'Reacher' in env_name else 3
        base_env = HIGLtoGCHRWrapper(base_env, env_name, goal_dim=goal_dim)
    
    return base_env


def make_env(args):
    """
    Unified environment creation function for GCHR framework.
    Supports both Gym (Fetch/Hand) and HIGL (PointMaze/AntMaze/Reacher/Pusher) environments.
    
    Args:
        args: Arguments object with env_name and other configuration
    
    Returns:
        Environment instance compatible with GCHR framework
    """
    env_name = args.env_name
    
    # Dictionary mapping for Gym environments (Fetch and Hand)
    gym_env_map = {
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
    
    # Check if it's a Gym environment
    if env_name in gym_env_map:
        try:
            env = gym.make(gym_env_map[env_name])
            
            # Set max episode steps
            if hasattr(env, '_max_episode_steps'):
                args.max_episode_steps = env._max_episode_steps
            else:
                args.max_episode_steps = 50
            
            print(f"[info] Created Gym environment: {gym_env_map[env_name]}")
            return env
            
        except Exception as e:
            raise Exception(f"[error] Failed to create Gym environment {env_name}: {str(e)}")
    
    # Check if it's a HIGL maze environment
    elif 'Maze' in env_name or 'Point' in env_name or 'Ant' in env_name:
        if not HIGL_AVAILABLE:
            raise Exception(f"[error] HIGL environments not available for {env_name}")
        
        # Determine maze size scaling from version
        if '-v0' in env_name:
            maze_size_scaling = 8
        elif '-v1' in env_name:
            maze_size_scaling = 4
        elif '-v2' in env_name:
            maze_size_scaling = 2
        else:
            maze_size_scaling = 4  # default
        
        reward_shaping = getattr(args, 'reward_shaping', 'sparse')
        evaluate = getattr(args, 'evaluate', False)
        
        env = create_higl_maze_env(
            env_name=env_name,
            reward_shaping=reward_shaping,
            maze_size_scaling=maze_size_scaling,
            evaluate=evaluate,
        )
        
        args.max_episode_steps = 500  # HIGL mazes use 500 steps
        print(f"[info] Created HIGL maze environment: {env_name}")
        return env
    
    # Check if it's a HIGL manipulator environment
    elif 'Reacher' in env_name or 'Pusher' in env_name:
        if not HIGL_AVAILABLE:
            raise Exception(f"[error] HIGL environments not available for {env_name}")
        
        reward_shaping = getattr(args, 'reward_shaping', 'dense')
        evaluate = getattr(args, 'evaluate', False)
        
        env = create_higl_manipulator_env(
            env_name=env_name,
            reward_shaping=reward_shaping,
            evaluate=evaluate,
        )
        
        args.max_episode_steps = 100  # HIGL manipulators use 100 steps
        print(f"[info] Created HIGL manipulator environment: {env_name}")
        return env
    
    else:
        raise Exception(f"[error] Unknown environment name: {env_name}")


def setup(args, env):
    """
    Setup function to initialize dimensions and parameters from environment.
    This is compatible with the original GCHR setup function.
    
    Args:
        args: Arguments object to populate with environment information
        env: Environment instance
    """
    obs = env.reset()
    
    # Handle both dict and array observations
    if isinstance(obs, dict):
        o = obs['observation']
        ag = obs['achieved_goal']
        g = obs['desired_goal']
    else:
        o = obs
        ag = obs[:2]  # assume first 2 dims are achieved goal
        g = obs[:2]   # placeholder
    
    # Set dimensions
    args.dim_state = o.shape[0] if hasattr(o, 'shape') else len(o)
    args.dim_goal = ag.shape[0] if hasattr(ag, 'shape') else len(ag)
    args.dim_action = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    
    print(f"[info] dim_state: {args.dim_state}")
    print(f"[info] dim_goal: {args.dim_goal}")
    print(f"[info] dim_action: {args.dim_action}")
    print(f"[info] max_action: {args.max_action}")
    
    # Find goal indices in observation
    # For HIGL environments, achieved_goal is typically the first goal_dim elements
    start_idx = None
    
    # Try to find where achieved_goal appears in observation
    if isinstance(obs, dict):
        # For dict observations, try to match achieved_goal in observation
        for i in range(args.dim_state - args.dim_goal + 1):
            sub_o = o[i:i+args.dim_goal]
            if np.allclose(sub_o, ag, atol=1e-6):
                start_idx = i
                break
    
    # If not found or not dict, assume it's at the beginning
    if start_idx is None:
        start_idx = 0
        print(f"[warning] Could not find achieved_goal in observation, assuming indices 0:{args.dim_goal}")
    
    print(f"[info] goal_idx starts at: {start_idx}")
    
    # Set goal indices
    import torch
    args.goal_idx = torch.arange(start_idx, start_idx + args.dim_goal)
    print(f"[info] goal_idx: {args.goal_idx}")
    
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Set experiment name
    suffix = "(+)rew" if not args.negative_reward else "(-)rew"
    if args.agent in ["her", "mher", "ddpg", "wgcsl", "dsim", "sac", "sac_her", "td_infonce", "gchr"]:
        if hasattr(args, 'critic'):
            suffix += f"_{args.critic}"
            if args.critic != "monolithic":
                suffix += f"_emb{args.dim_embed}"
        if args.terminate:
            suffix += "_terminate"
    
    args.experiment_name = f"{args.env_name}_{args.agent}_{suffix}_lr{getattr(args, 'lr_critic', args.lr_actor)}_sd{args.seed}"
    
    print(f"[info] Experiment name: {args.experiment_name}")
    print(f"[info] Max episode steps: {args.max_episode_steps}")


# Default environment parameters for HIGL environments
HIGL_DEFAULT_ENV_PARAMS = {
    'PointMaze-v1': {
        'wgcsl_baw_delta': 0.15,
    },
    'AntMaze-v1': {
        'wgcsl_baw_delta': 0.15,
    },
    'PointMaze-v0': {
        'wgcsl_baw_delta': 0.15,
    },
    'AntMaze-v0': {
        'wgcsl_baw_delta': 0.15,
    },
    'Reacher3D-v0': {
        'wgcsl_baw_delta': 0.15,
    },
    'Pusher-v0': {
        'wgcsl_baw_delta': 0.15,
    },
}


def update_args_with_env_params(args, env_params=None):
    """
    Update args with environment-specific parameters.
    
    Args:
        args: Arguments object
        env_params: Dictionary of environment parameters (optional)
    """
    if env_params is None:
        env_params = HIGL_DEFAULT_ENV_PARAMS
    
    env_name = args.env_name
    
    if env_name in env_params:
        for k, v in env_params[env_name].items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)
                print(f"[info] Set {k} = {v} for {env_name}")