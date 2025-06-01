"""
PettingZoo environment wrapper for the chase game.
"""

import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector

from board_state import BoardState


class ChaseGameEnv(AECEnv):
    """PettingZoo AEC environment for the chase game."""
    
    metadata = {
        "render_modes": ["human"],
        "name": "chase_game_v0",
    }
    
    def __init__(self, board_size: int = 15, max_turns: int = 300, seed: int = None, render_mode: str = None):
        """Initialize the environment."""
        super().__init__()
        
        self.board_size = board_size
        self.max_turns = max_turns
        self._seed = seed
        self.render_mode = render_mode
        
        # Agent names
        self.possible_agents = ["runner", "chaser1", "chaser2"]
        
        # Action and observation spaces
        self._action_spaces = {
            agent: spaces.Discrete(5) for agent in self.possible_agents
        }
        
        # Observation space: dict with board and positions
        self._observation_spaces = {
            agent: spaces.Dict({
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
            }) for agent in self.possible_agents
        }
        
        # Initialize board state
        self.board_state = BoardState(
            board_size=board_size,
            max_turns=max_turns,
            seed=seed
        )
        
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            self._seed = seed
            
        # Reset board state
        self.board_state = BoardState(
            board_size=self.board_size,
            max_turns=self.max_turns,
            seed=self._seed
        )
        self.board_state.reset()
        
        # Reset agent tracking
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Set up agent selector
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
    def step(self, action):
        """Process one agent's action."""
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # If agent is done, automatically skip
            self._was_dead_step(action)
            return
            
        # Current agent takes action
        agent = self.agent_selection
        
        # Attempt move
        valid_move = self.board_state.move_agent(agent, action)
        
        if not valid_move:
            # Invalid move - this shouldn't happen with action masking
            # but handle gracefully
            pass
        
        # Check game status and assign rewards
        if self.board_state.is_terminal():
            if self.board_state.winner == "runner":
                # Runner wins
                self.rewards = {
                    "runner": 1,
                    "chaser1": 0,
                    "chaser2": 0
                }
            else:
                # Chasers win
                self.rewards = {
                    "runner": -1,
                    "chaser1": 1,
                    "chaser2": 1
                }
            
            # Mark all agents as terminated
            self.terminations = {agent: True for agent in self.agents}
        else:
            # No rewards during regular play
            self.rewards = {agent: 0 for agent in self.agents}
        
        # Accumulate rewards
        self._accumulate_rewards()
        
        # Move to next agent
        self.agent_selection = self._agent_selector.next()
        
    def observe(self, agent):
        """Get observation for an agent."""
        obs = self.board_state.get_observation()
        
        # Convert positions to numpy arrays
        obs["runner_pos"] = np.array(obs["runner_pos"], dtype=np.int32)
        obs["chaser1_pos"] = np.array(obs["chaser1_pos"], dtype=np.int32)
        obs["chaser2_pos"] = np.array(obs["chaser2_pos"], dtype=np.int32)
        
        # Action mask for current agent
        if agent == self.agent_selection:
            legal_moves = self.board_state.get_legal_moves(agent)
            action_mask = np.zeros(5, dtype=np.int8)
            for move in legal_moves:
                action_mask[move] = 1
            self.infos[agent]["action_mask"] = action_mask
        
        # Add other info
        self.infos[agent].update({
            "turn_number": self.board_state.turn_count,
            "legal_moves": self.board_state.get_legal_moves(agent),
            "win_condition": self._get_win_condition()
        })
        
        return obs
    
    def render(self):
        """Render the game state."""
        if self.render_mode == "human":
            from utils import render_board, format_game_status
            print(render_board(self.board_state, clear_screen=False))
            print(format_game_status(self.board_state))
    
    def close(self):
        """Close the environment."""
        pass
    
    def _get_win_condition(self):
        """Get current win condition status."""
        if not self.board_state.is_terminal():
            return None
        elif self.board_state.winner == "runner":
            return "runner_win"
        else:
            return "chaser_win"
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """Get observation space for an agent."""
        return self._observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """Get action space for an agent."""
        return self._action_spaces[agent]