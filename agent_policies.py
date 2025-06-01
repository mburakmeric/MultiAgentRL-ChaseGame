"""
Agent policy implementations for the chase game.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import random


class BaseAgent(ABC):
    """Base class for all agent policies."""
    
    def __init__(self, name: str):
        """Initialize agent with a name."""
        self.name = name
    
    @abstractmethod
    def decide_move(self, observation: Dict[str, Any], legal_moves: List[int]) -> int:
        """
        Decide which move to make based on observation and legal moves.
        
        Args:
            observation: Dictionary containing board state and positions
            legal_moves: List of legal action indices
            
        Returns:
            Selected action index
        """
        pass


class RandomAgent(BaseAgent):
    """Agent that selects moves uniformly at random from legal moves."""
    
    def decide_move(self, observation: Dict[str, Any], legal_moves: List[int]) -> int:
        """Select a random legal move."""
        return random.choice(legal_moves)


class ManualAgent(BaseAgent):
    """Agent that takes user input for moves."""
    
    # Action mapping
    ACTION_NAMES = {
        0: "UP",
        1: "DOWN", 
        2: "LEFT",
        3: "RIGHT",
        4: "STAY"
    }
    
    def decide_move(self, observation: Dict[str, Any], legal_moves: List[int]) -> int:
        """Get move from user input."""
        # Display legal moves
        move_names = [self.ACTION_NAMES[move] for move in legal_moves]
        print(f"\n{self.name}'s turn. Legal moves: {', '.join(move_names)}")
        
        while True:
            try:
                # Get user input
                user_input = input("Enter move (U/D/L/R/S): ").strip().upper()
                
                # Map input to action
                input_map = {
                    'U': 0,  # UP
                    'D': 1,  # DOWN
                    'L': 2,  # LEFT
                    'R': 3,  # RIGHT
                    'S': 4   # STAY
                }
                
                if user_input not in input_map:
                    print("Invalid input. Use U/D/L/R/S.")
                    continue
                    
                action = input_map[user_input]
                
                if action not in legal_moves:
                    print(f"Move {self.ACTION_NAMES[action]} is not legal. Try again.")
                    continue
                    
                return action
                
            except KeyboardInterrupt:
                print("\nGame interrupted by user.")
                raise
            except Exception as e:
                print(f"Error reading input: {e}. Try again.")