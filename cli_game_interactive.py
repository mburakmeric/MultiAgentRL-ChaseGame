"""
Interactive terminal game runner for the chase game.
Implements the prompt-based interface as specified in termproject.txt.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from board_state import BoardState, Position
from agent_policies import RandomAgent, ManualAgent, BaseAgent
from utils import render_board, format_game_status


def parse_position(pos_str: str) -> Tuple[int, int]:
    """
    Parse position string like 'a 3' to board coordinates.
    Returns (row, col) in 0-indexed format.
    """
    parts = pos_str.strip().split()
    if len(parts) != 2:
        raise ValueError("Position must be in format 'column row' (e.g., 'a 3')")
    
    col_letter = parts[0].lower()
    row_str = parts[1]
    
    # Convert column letter to index (a=0, b=1, etc.)
    if len(col_letter) != 1 or not 'a' <= col_letter <= 'o':
        raise ValueError(f"Column must be a-o, got '{col_letter}'")
    col = ord(col_letter) - ord('a')
    
    # Convert row to index (1-based to 0-based)
    try:
        row = int(row_str) - 1
        if not 0 <= row < 15:
            raise ValueError(f"Row must be 1-15, got {row_str}")
    except ValueError:
        raise ValueError(f"Row must be a number 1-15, got '{row_str}'")
    
    return (row, col)


def get_valid_position(prompt: str, board: np.ndarray, existing_positions: List[Position] = None) -> Position:
    """Get a valid position from user input."""
    while True:
        try:
            pos_str = input(prompt)
            row, col = parse_position(pos_str)
            
            # Check if position is empty
            if board[row, col] != BoardState.EMPTY:
                print(f"Position {pos_str} is already occupied. Please choose an empty position.")
                continue
            
            # Check distance constraints if placing agents
            if existing_positions:
                new_pos = Position(row, col)
                valid = True
                for existing_pos in existing_positions:
                    if new_pos.manhattan_distance(existing_pos) < 7:
                        print(f"Position too close to existing agent (Manhattan distance must be >= 7)")
                        valid = False
                        break
                if not valid:
                    continue
            
            return Position(row, col)
            
        except ValueError as e:
            print(f"Invalid position: {e}")


def get_obstacle_positions(num_obstacles: int, board: np.ndarray, 
                         agent_positions: List[Position]) -> List[Position]:
    """Get obstacle positions, checking Chebyshev distance from agents."""
    obstacles = []
    
    for i in range(num_obstacles):
        while True:
            try:
                pos_str = input(f"Enter the position of obstacle {i+1}: ")
                row, col = parse_position(pos_str)
                
                # Check if position is empty
                if board[row, col] != BoardState.EMPTY:
                    print(f"Position {pos_str} is already occupied.")
                    continue
                
                # Check Chebyshev distance from agents
                new_pos = Position(row, col)
                too_close = False
                for agent_pos in agent_positions:
                    if new_pos.chebyshev_distance(agent_pos) <= 1:
                        print(f"Obstacle too close to agent (Chebyshev distance must be > 1)")
                        too_close = True
                        break
                
                if not too_close:
                    obstacles.append(new_pos)
                    board[row, col] = BoardState.OBSTACLE
                    break
                    
            except ValueError as e:
                print(f"Invalid position: {e}")
    
    return obstacles


def setup_board_interactive() -> Tuple[BoardState, Dict[str, str]]:
    """
    Interactively set up the game board and return configured BoardState and agent policies.
    """
    print("=== CHASE GAME SETUP ===\n")
    
    # Get number of turns
    while True:
        try:
            max_turns = int(input("Enter the number of turns: "))
            if max_turns <= 0:
                print("Number of turns must be positive.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Get number of obstacles
    while True:
        try:
            num_obstacles = int(input("Enter the number of black cells on the board: "))
            if num_obstacles < 0:
                print("Number of obstacles cannot be negative.")
                continue
            if num_obstacles > 200:  # Reasonable upper limit
                print("Too many obstacles. Please enter a smaller number.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Create board state with custom settings
    board_state = BoardState(num_obstacles=num_obstacles, max_turns=max_turns)
    
    # Get obstacle placement method
    while True:
        placement = input("Are the cells going to be determined randomly or manually [r/m]: ").lower()
        if placement in ['r', 'm']:
            break
        print("Please enter 'r' for random or 'm' for manual.")
    
    # Get agent positions first
    print("\nEnter agent positions:")
    temp_board = np.zeros((15, 15), dtype=int)  # Temporary board for validation
    
    runner_pos = get_valid_position("Enter the position of Runner: ", temp_board)
    temp_board[runner_pos.x, runner_pos.y] = BoardState.RUNNER
    
    chaser1_pos = get_valid_position("Enter the position of Chaser1: ", temp_board, [runner_pos])
    temp_board[chaser1_pos.x, chaser1_pos.y] = BoardState.CHASER1
    
    chaser2_pos = get_valid_position("Enter the position of Chaser2: ", temp_board, [runner_pos, chaser1_pos])
    temp_board[chaser2_pos.x, chaser2_pos.y] = BoardState.CHASER2
    
    custom_positions = {
        'runner': runner_pos,
        'chaser1': chaser1_pos,
        'chaser2': chaser2_pos
    }
    
    # Handle obstacles
    custom_obstacles = None
    if placement == 'm' and num_obstacles > 0:
        print("\nEnter obstacle positions:")
        custom_obstacles = get_obstacle_positions(
            num_obstacles, temp_board, 
            [runner_pos, chaser1_pos, chaser2_pos]
        )
    
    # Reset board with custom positions and obstacles
    if placement == 'r':
        board_state.reset(custom_positions=custom_positions)
    else:
        board_state.reset(custom_positions=custom_positions, custom_obstacles=custom_obstacles)
    
    # Get agent control methods
    print("\nAgent control setup:")
    
    # Runner control
    print("\nHow should the Runner be controlled?")
    print("1. Random")
    print("2. Manual (user input)")
    print("3. Neural network/function (not implemented yet)")
    
    while True:
        runner_choice = input("Select Runner control method [1/2/3]: ")
        if runner_choice == '1':
            runner_policy = 'random'
            break
        elif runner_choice == '2':
            runner_policy = 'manual'
            break
        elif runner_choice == '3':
            print("Neural network control not yet implemented. Defaulting to random.")
            runner_policy = 'random'
            break
        else:
            print("Please enter 1, 2, or 3.")
    
    # For chasers, we'll default to random for now
    chaser1_policy = 'random'
    chaser2_policy = 'random'
    
    print(f"\nChaser1 will be controlled by: random policy")
    print(f"Chaser2 will be controlled by: random policy")
    
    policies = {
        'runner': runner_policy,
        'chaser1': chaser1_policy,
        'chaser2': chaser2_policy
    }
    
    return board_state, policies


def play_interactive_game():
    """Play a game with interactive setup."""
    # Set up the game
    board_state, policies = setup_board_interactive()
    
    # Create agents based on policies
    agents: Dict[str, BaseAgent] = {}
    for agent_name, policy in policies.items():
        if policy == "random":
            agents[agent_name] = RandomAgent(agent_name)
        elif policy == "manual":
            agents[agent_name] = ManualAgent(agent_name)
    
    # Game loop
    print("\n=== GAME START ===\n")
    
    while not board_state.is_terminal():
        # Render board
        print(render_board(board_state, clear_screen=False))
        print(format_game_status(board_state))
        
        # Get current agent
        current_agent_name = board_state.get_current_agent()
        current_agent = agents[current_agent_name]
        
        # Get observation and legal moves
        observation = board_state.get_observation()
        legal_moves = board_state.get_legal_moves(current_agent_name)
        
        # Agent decides move
        action = current_agent.decide_move(observation, legal_moves)
        
        # Execute move
        board_state.move_agent(current_agent_name, action)
        
        # Small pause for visibility (only for random agents)
        if isinstance(current_agent, RandomAgent):
            import time
            time.sleep(0.3)
    
    # Game over - show final state
    print("\n" + "="*30)
    print(render_board(board_state, clear_screen=False))
    print(format_game_status(board_state))
    print("="*30)
    
    return board_state.winner


def main():
    """Main entry point for interactive CLI."""
    try:
        play_interactive_game()
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()