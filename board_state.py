"""
Core board logic for the chase game.
Handles grid state, spawning, movement, and game rules.
"""

import numpy as np
from typing import Tuple, List, Optional, Set
from dataclasses import dataclass
import random


@dataclass
class Position:
    """Represents a position on the board."""
    x: int
    y: int
    
    def manhattan_distance(self, other: 'Position') -> int:
        """Calculate Manhattan distance to another position."""
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def chebyshev_distance(self, other: 'Position') -> int:
        """Calculate Chebyshev distance to another position."""
        return max(abs(self.x - other.x), abs(self.y - other.y))


class BoardState:
    """Manages the game board state and logic."""
    
    # Board encoding constants
    EMPTY = 0
    OBSTACLE = 1
    RUNNER = 2
    CHASER1 = 3
    CHASER2 = 4
    
    # Action constants
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    
    def __init__(self, board_size: int = 15, num_obstacles: int = 5, max_turns: int = 300, seed: Optional[int] = None):
        """Initialize the board state."""
        self.board_size = board_size
        self.num_obstacles = num_obstacles
        self.max_turns = max_turns
        self.turn_count = 0
        self.current_agent_idx = 0  # 0: runner, 1: chaser1, 2: chaser2
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize board and positions
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.runner_pos = None
        self.chaser1_pos = None
        self.chaser2_pos = None
        
        # Game state
        self.game_over = False
        self.winner = None  # "runner" or "chasers"
        
    def reset(self) -> None:
        """Reset the board to initial state."""
        self.board.fill(self.EMPTY)
        self.turn_count = 0
        self.current_agent_idx = 0
        self.game_over = False
        self.winner = None
        
        # Place agents
        self._place_agents()
        
        # Place obstacles
        self._place_obstacles()
        
    def _place_agents(self) -> None:
        """Place agents on the board with distance constraints."""
        # Generate all possible positions
        all_positions = [Position(x, y) for x in range(self.board_size) for y in range(self.board_size)]
        
        # Place runner first
        runner_idx = random.randint(0, len(all_positions) - 1)
        self.runner_pos = all_positions[runner_idx]
        self.board[self.runner_pos.x, self.runner_pos.y] = self.RUNNER
        
        # Find valid positions for chasers (Manhattan distance >= 7 from runner)
        valid_chaser_positions = [
            pos for pos in all_positions 
            if pos.manhattan_distance(self.runner_pos) >= 7
        ]
        
        if len(valid_chaser_positions) < 2:
            raise ValueError("Not enough valid positions for chasers with distance constraint")
        
        # Place chasers
        chaser_positions = random.sample(valid_chaser_positions, 2)
        self.chaser1_pos = chaser_positions[0]
        self.chaser2_pos = chaser_positions[1]
        
        self.board[self.chaser1_pos.x, self.chaser1_pos.y] = self.CHASER1
        self.board[self.chaser2_pos.x, self.chaser2_pos.y] = self.CHASER2
        
    def _place_obstacles(self) -> None:
        """Place obstacles on the board, avoiding agent spawn zones."""
        # Compute forbidden zones (Chebyshev distance <= 1 from each agent)
        forbidden_positions = set()
        
        for agent_pos in [self.runner_pos, self.chaser1_pos, self.chaser2_pos]:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    x, y = agent_pos.x + dx, agent_pos.y + dy
                    if 0 <= x < self.board_size and 0 <= y < self.board_size:
                        forbidden_positions.add((x, y))
        
        # Find all valid obstacle positions
        valid_obstacle_positions = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[x, y] == self.EMPTY and (x, y) not in forbidden_positions:
                    valid_obstacle_positions.append((x, y))
        
        if len(valid_obstacle_positions) < self.num_obstacles:
            raise ValueError("Not enough valid positions for obstacles")
        
        # Place obstacles
        obstacle_positions = random.sample(valid_obstacle_positions, self.num_obstacles)
        for x, y in obstacle_positions:
            self.board[x, y] = self.OBSTACLE
            
    def get_current_agent(self) -> str:
        """Get the name of the current agent to move."""
        return ["runner", "chaser1", "chaser2"][self.current_agent_idx]
    
    def get_agent_position(self, agent: str) -> Position:
        """Get the position of a specific agent."""
        if agent == "runner":
            return self.runner_pos
        elif agent == "chaser1":
            return self.chaser1_pos
        elif agent == "chaser2":
            return self.chaser2_pos
        else:
            raise ValueError(f"Unknown agent: {agent}")
            
    def get_legal_moves(self, agent: str) -> List[int]:
        """Get list of legal moves for an agent."""
        pos = self.get_agent_position(agent)
        legal_moves = []
        
        # Check each direction
        directions = [
            (self.UP, -1, 0),
            (self.DOWN, 1, 0),
            (self.LEFT, 0, -1),
            (self.RIGHT, 0, 1)
        ]
        
        for action, dx, dy in directions:
            new_x, new_y = pos.x + dx, pos.y + dy
            
            # Check bounds
            if 0 <= new_x < self.board_size and 0 <= new_y < self.board_size:
                # Check for obstacles
                if self.board[new_x, new_y] != self.OBSTACLE:
                    legal_moves.append(action)
        
        # Stay is only legal if no other moves are available
        if not legal_moves:
            legal_moves.append(self.STAY)
            
        return legal_moves
    
    def move_agent(self, agent: str, action: int) -> bool:
        """Move an agent and check for game end conditions. Returns True if move was valid."""
        if self.game_over:
            return False
            
        # Verify it's this agent's turn
        expected_agent = self.get_current_agent()
        if agent != expected_agent:
            return False
            
        # Check if action is legal
        legal_moves = self.get_legal_moves(agent)
        if action not in legal_moves:
            return False
            
        # Get current position
        pos = self.get_agent_position(agent)
        
        # Clear current position on board
        self.board[pos.x, pos.y] = self.EMPTY
        
        # Calculate new position
        new_x, new_y = pos.x, pos.y
        if action == self.UP:
            new_x -= 1
        elif action == self.DOWN:
            new_x += 1
        elif action == self.LEFT:
            new_y -= 1
        elif action == self.RIGHT:
            new_y += 1
        # STAY: position remains the same
        
        # Update position
        new_pos = Position(new_x, new_y)
        if agent == "runner":
            self.runner_pos = new_pos
            # Update board (handle potential overlap with chasers)
            if self.board[new_x, new_y] in [self.CHASER1, self.CHASER2]:
                # Runner caught!
                self.game_over = True
                self.winner = "chasers"
            else:
                self.board[new_x, new_y] = self.RUNNER
        elif agent == "chaser1":
            self.chaser1_pos = new_pos
            # Check if catching runner
            if self.board[new_x, new_y] == self.RUNNER:
                self.game_over = True
                self.winner = "chasers"
            else:
                # Chasers can overlap
                if self.board[new_x, new_y] != self.CHASER2:
                    self.board[new_x, new_y] = self.CHASER1
        else:  # chaser2
            self.chaser2_pos = new_pos
            # Check if catching runner
            if self.board[new_x, new_y] == self.RUNNER:
                self.game_over = True
                self.winner = "chasers"
            else:
                # Chasers can overlap
                if self.board[new_x, new_y] != self.CHASER1:
                    self.board[new_x, new_y] = self.CHASER2
        
        # Check for runner win condition (surrounded)
        if not self.game_over and agent == "runner":
            runner_moves = self.get_legal_moves("runner")
            if runner_moves == [self.STAY]:
                # Runner is surrounded and no chaser is on their position
                if self.board[self.runner_pos.x, self.runner_pos.y] == self.RUNNER:
                    self.game_over = True
                    self.winner = "runner"
        
        # Advance to next agent
        self.current_agent_idx = (self.current_agent_idx + 1) % 3
        
        # If we've completed a full round, increment turn count
        if self.current_agent_idx == 0:
            self.turn_count += 1
            
            # Check for turn limit
            if self.turn_count >= self.max_turns:
                self.game_over = True
                self.winner = "runner"
        
        return True
    
    def is_terminal(self) -> bool:
        """Check if the game has ended."""
        return self.game_over
    
    def get_observation(self) -> dict:
        """Get the full observation of the current state."""
        return {
            "board": self.board.copy(),
            "runner_pos": (self.runner_pos.x, self.runner_pos.y),
            "chaser1_pos": (self.chaser1_pos.x, self.chaser1_pos.y),
            "chaser2_pos": (self.chaser2_pos.x, self.chaser2_pos.y)
        }