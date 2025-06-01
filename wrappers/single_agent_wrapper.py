"""
Wrapper to convert PettingZoo environment to single-agent perspective for SB3 training.
"""

import numpy as np
from gymnasium import spaces
import gymnasium as gym
from typing import Dict, Any, Optional, Tuple

from cgame_env import ChaseGameEnv
from agent_policies import BaseAgent, RandomAgent


class SingleAgentWrapper(gym.Env):
    """
    Wraps the PettingZoo Chase Game environment to train a single agent.
    Other agents use fixed policies (random or pre-trained).
    """
    
    def __init__(
        self, 
        agent_to_train: str,
        opponent_policies: Dict[str, BaseAgent],
        board_size: int = 15,
        max_turns: int = 300,
        seed: Optional[int] = None
    ):
        """
        Initialize the single-agent wrapper.
        
        Args:
            agent_to_train: Name of the agent to train ("runner", "chaser1", or "chaser2")
            opponent_policies: Dictionary mapping other agent names to their policies
            board_size: Size of the game board
            max_turns: Maximum number of turns
            seed: Random seed
        """
        super().__init__()
        
        self.agent_to_train = agent_to_train
        self.opponent_policies = opponent_policies
        
        # Create the underlying PettingZoo environment
        self.env = ChaseGameEnv(
            board_size=board_size,
            max_turns=max_turns,
            seed=seed
        )
        
        # Set up observation and action spaces
        self.observation_space = spaces.Dict({
            "board": spaces.Box(
                low=0, high=4, shape=(board_size, board_size), dtype=np.int32
            ),
            "runner_pos": spaces.Box(
                low=0, high=board_size-1, shape=(2,), dtype=np.int32
            ),
            "chaser1_pos": spaces.Box(
                low=0, high=board_size-1, shape=(2,), dtype=np.int32
            ),
            "chaser2_pos": spaces.Box(
                low=0, high=board_size-1, shape=(2,), dtype=np.int32
            ),
        })
        
        self.action_space = spaces.Discrete(5)
        
        # Track episode state
        self._last_obs = None
        self._action_mask = None
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment."""
        # Reset PettingZoo environment
        self.env.reset(seed=seed)
        
        # Get initial observation for our agent
        self._last_obs = None
        self._action_mask = None
        
        # Step through agents until we reach the one we're training
        for agent in self.env.agent_iter():
            obs, reward, termination, truncation, info = self.env.last()
            
            if agent == self.agent_to_train:
                self._last_obs = obs
                self._action_mask = info.get("action_mask", np.ones(5, dtype=np.int8))
                break
            else:
                # Other agents take their policy actions
                if termination or truncation:
                    action = None
                else:
                    policy = self.opponent_policies[agent]
                    legal_moves = info.get("legal_moves", list(range(5)))
                    action = policy.decide_move(obs, legal_moves)
                
                self.env.step(action)
        
        return self._last_obs, {"action_mask": self._action_mask}
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Take a step in the environment."""
        # Take the action for our training agent
        self.env.step(action)
        
        # Initialize tracking variables
        episode_done = False
        episode_reward = 0.0
        
        # Continue stepping through other agents
        for agent in self.env.agent_iter():
            obs, reward, termination, truncation, info = self.env.last()
            
            # Check if episode is done
            if termination or truncation:
                episode_done = True
                if agent == self.agent_to_train:
                    episode_reward = reward
            
            if agent == self.agent_to_train:
                # Our turn again
                self._last_obs = obs
                self._action_mask = info.get("action_mask", np.ones(5, dtype=np.int8))
                
                # If episode is done, return immediately
                if episode_done:
                    return obs, episode_reward, termination, truncation, {
                        "action_mask": self._action_mask,
                        "win_condition": info.get("win_condition", None)
                    }
                
                # Episode continues, return and wait for next action
                return obs, 0.0, False, False, {"action_mask": self._action_mask}
            else:
                # Other agents take their actions
                if not (termination or truncation):
                    policy = self.opponent_policies[agent]
                    legal_moves = info.get("legal_moves", list(range(5)))
                    action = policy.decide_move(obs, legal_moves)
                else:
                    action = None
                
                self.env.step(action)
        
        # Should not reach here
        return self._last_obs, 0.0, False, False, {"action_mask": self._action_mask}
    
    def render(self):
        """Render the environment."""
        self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()


def make_training_env(
    agent_to_train: str,
    opponent_policy_types: Dict[str, str] = None,
    **env_kwargs
) -> SingleAgentWrapper:
    """
    Create a training environment with specified opponent policies.
    
    Args:
        agent_to_train: Agent to train ("runner", "chaser1", or "chaser2")
        opponent_policy_types: Dict mapping agent names to policy types ("random", etc.)
        **env_kwargs: Additional arguments for the environment
        
    Returns:
        SingleAgentWrapper environment
    """
    # Default to random opponents
    if opponent_policy_types is None:
        opponent_policy_types = {
            "runner": "random",
            "chaser1": "random",
            "chaser2": "random"
        }
        opponent_policy_types.pop(agent_to_train)
    
    # Create opponent policies
    opponent_policies = {}
    for agent_name, policy_type in opponent_policy_types.items():
        if agent_name != agent_to_train:
            if policy_type == "random":
                opponent_policies[agent_name] = RandomAgent(agent_name)
            else:
                raise ValueError(f"Unknown policy type: {policy_type}")
    
    return SingleAgentWrapper(
        agent_to_train=agent_to_train,
        opponent_policies=opponent_policies,
        **env_kwargs
    )