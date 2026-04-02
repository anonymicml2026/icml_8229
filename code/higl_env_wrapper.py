"""
HIGL Environment Wrapper for GCHR Framework
This wrapper adapts HIGL environments (PointMaze, AntMaze, Reacher, Pusher) 
to be compatible with the GCHR framework's expected observation format.
"""

import numpy as np
import gym
from collections import OrderedDict


class HIGLtoGCHRWrapper(gym.Wrapper):
    """
    Wrapper that converts HIGL environment observations to GCHR format.
    
    HIGL environments return observations as:
    - PointMaze/AntMaze: raw observation array
    - Reacher/Pusher: dict with 'observation', 'desired_goal', 'achieved_goal'
    
    GCHR expects observations as dict with:
    - 'observation': full state
    - 'achieved_goal': current achieved goal (usually first 2 or 3 dims)
    - 'desired_goal': target goal
    """
    
    def __init__(self, env, env_name, goal_dim=2):
        super(HIGLtoGCHRWrapper, self).__init__(env)
        self.env_name = env_name
        self.goal_dim = goal_dim
        
        # Determine if this is a maze environment or manipulator environment
        self.is_maze = 'Maze' in env_name or 'Point' in env_name or 'Ant' in env_name
        self.is_manipulator = 'Reacher' in env_name or 'Pusher' in env_name
        
        # For maze environments, we need to track the goal
        self.current_goal = None
        self.distance_threshold = 5.0  # default for maze environments
        
        # Update observation space to dict format
        if self.is_maze:
            obs_space = env.observation_space
            goal_space = gym.spaces.Box(
                low=-np.inf * np.ones(self.goal_dim),
                high=np.inf * np.ones(self.goal_dim),
                dtype=np.float32
            )
            self.observation_space = gym.spaces.Dict(OrderedDict({
                'observation': obs_space,
                'achieved_goal': goal_space,
                'desired_goal': goal_space,
            }))
    
    def reset(self):
        """Reset environment and return GCHR-compatible observation."""
        obs = self.env.reset()
        
        if self.is_maze:
            # For maze environments, obs is a numpy array
            # achieved_goal is the first goal_dim elements (x, y position)
            achieved_goal = obs[:self.goal_dim].copy()
            
            # Sample a random goal if not set by the environment
            if self.current_goal is None:
                if 'AntMaze-v1' in self.env_name or 'PointMaze-v1' in self.env_name:
                    # Sample goal in valid maze region
                    self.current_goal = np.random.uniform((-2, -2), (10, 10))
                elif 'AntMaze-v0' in self.env_name or 'PointMaze-v0' in self.env_name:
                    self.current_goal = np.random.uniform((-4, -4), (20, 20))
                else:
                    self.current_goal = np.random.uniform((-5, -5), (5, 5))
            
            return {
                'observation': obs.copy(),
                'achieved_goal': achieved_goal,
                'desired_goal': self.current_goal.copy(),
            }
        
        elif self.is_manipulator:
            # For manipulator environments, obs is already a dict
            if isinstance(obs, dict):
                return obs
            else:
                # If it's not a dict, create one
                achieved_goal = obs[:self.goal_dim].copy()
                desired_goal = np.zeros(self.goal_dim)  # placeholder
                return {
                    'observation': obs.copy(),
                    'achieved_goal': achieved_goal,
                    'desired_goal': desired_goal,
                }
        
        else:
            raise ValueError(f"Unknown environment type: {self.env_name}")
    
    def step(self, action):
        """Step environment and return GCHR-compatible observation."""
        obs, reward, done, info = self.env.step(action)
        
        if self.is_maze:
            # For maze environments
            achieved_goal = obs[:self.goal_dim].copy()
            
            # Compute reward based on distance to goal
            if self.current_goal is not None:
                distance = np.linalg.norm(achieved_goal - self.current_goal)
                # Step-style reward: 0 if within threshold, -1 otherwise
                reward = -(distance > self.distance_threshold).astype(np.float32)
                info['is_success'] = (distance <= self.distance_threshold)
            
            next_obs = {
                'observation': obs.copy(),
                'achieved_goal': achieved_goal,
                'desired_goal': self.current_goal.copy(),
            }
            
            return next_obs, reward, done, info
        
        elif self.is_manipulator:
            # For manipulator environments, obs should already be a dict
            if isinstance(obs, dict):
                return obs, reward, done, info
            else:
                achieved_goal = obs[:self.goal_dim].copy()
                next_obs = {
                    'observation': obs.copy(),
                    'achieved_goal': achieved_goal,
                    'desired_goal': self.current_goal.copy() if self.current_goal is not None else achieved_goal,
                }
                return next_obs, reward, done, info
        
        else:
            raise ValueError(f"Unknown environment type: {self.env_name}")
    
    def set_goal(self, goal):
        """Manually set the goal for the environment."""
        self.current_goal = goal.copy()
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute reward for HER-style goal relabeling."""
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(distance > self.distance_threshold).astype(np.float32)


class EnvWithGoalWrapper(gym.Wrapper):
    """
    Enhanced wrapper that provides more control over goal sampling and rewards.
    This is similar to the EnvWithGoal class from HIGL but adapted for GCHR.
    """
    
    def __init__(self, base_env, env_name, fix_goal=False, manual_goals=None, 
                 step_style=True, evaluate=False):
        super(EnvWithGoalWrapper, self).__init__(base_env)
        
        self.env_name = env_name
        self.fix_goal = fix_goal
        self.manual_goals = manual_goals
        self.step_style = step_style
        self.evaluate = evaluate
        
        # Determine goal dimension
        if env_name in ['AntMaze-v1', 'PointMaze-v1', 'AntMaze-v0', 'PointMaze-v0']:
            self.goal_dim = 2
        elif env_name in ['Reacher3D-v0']:
            self.goal_dim = 3
        elif env_name in ['Pusher-v0']:
            self.goal_dim = 3
        else:
            self.goal_dim = 2  # default
        
        # Set distance threshold
        if env_name in ['AntMaze-v1', 'PointMaze-v1']:
            self.distance_threshold = 2.5
        elif env_name in ['AntMaze-v0', 'PointMaze-v0']:
            self.distance_threshold = 5.0
        else:
            self.distance_threshold = 0.25
        
        self.goal = None
        self.count = 0
        self.max_steps = 500
        
    def _get_goal_sample_fn(self):
        """Get goal sampling function based on environment."""
        if self.env_name in ['AntMaze-v1', 'PointMaze-v1']:
            if self.evaluate:
                return lambda: np.array([0., 8.])
            else:
                if self.fix_goal:
                    return lambda: np.array([0., 8.])
                else:
                    return lambda: np.random.uniform((-2, -2), (10, 10))
        
        elif self.env_name in ['AntMaze-v0', 'PointMaze-v0']:
            if self.evaluate:
                return lambda: np.array([0., 16.])
            else:
                return lambda: np.random.uniform((-4, -4), (20, 20))
        
        else:
            return lambda: np.random.uniform((-5, -5), (5, 5))
    
    def _compute_reward(self, achieved_goal, desired_goal):
        """Compute reward based on achieved and desired goal."""
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        
        if self.step_style:
            # Step-style: 0 if success, -1 otherwise
            return -(distance > self.distance_threshold).astype(np.float32)
        else:
            # Dense reward: negative distance
            return -distance
    
    def _is_success(self, achieved_goal, desired_goal):
        """Check if goal is achieved."""
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return distance <= self.distance_threshold
    
    def reset(self):
        """Reset environment."""
        obs = self.base_env.reset()
        self.count = 0
        
        # Sample new goal
        goal_sample_fn = self._get_goal_sample_fn()
        self.goal = goal_sample_fn()
        
        # Handle different observation formats
        if isinstance(obs, dict):
            achieved_goal = obs.get('achieved_goal', obs['observation'][:self.goal_dim])
            return {
                'observation': obs['observation'],
                'achieved_goal': achieved_goal,
                'desired_goal': self.goal.copy(),
            }
        else:
            # obs is a numpy array
            achieved_goal = obs[:self.goal_dim]
            return {
                'observation': obs,
                'achieved_goal': achieved_goal,
                'desired_goal': self.goal.copy(),
            }
    
    def step(self, action):
        """Step environment."""
        obs, _, done, info = self.base_env.step(action)
        self.count += 1
        
        # Handle different observation formats
        if isinstance(obs, dict):
            achieved_goal = obs.get('achieved_goal', obs['observation'][:self.goal_dim])
            observation = obs['observation']
        else:
            achieved_goal = obs[:self.goal_dim]
            observation = obs
        
        # Compute reward
        reward = self._compute_reward(achieved_goal, self.goal)
        
        # Check success
        is_success = self._is_success(achieved_goal, self.goal)
        info['is_success'] = is_success
        
        # Check done
        done = done or (self.count >= self.max_steps)
        
        next_obs = {
            'observation': observation,
            'achieved_goal': achieved_goal,
            'desired_goal': self.goal.copy(),
        }
        
        return next_obs, reward, done, info