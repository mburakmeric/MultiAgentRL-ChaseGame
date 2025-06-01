"""
Terminal game runner for the chase game.
Basic version with manual/random agent options.
"""

import argparse
from typing import Dict, Optional

from board_state import BoardState
from agent_policies import RandomAgent, ManualAgent, BaseAgent
from utils import render_board, format_game_status


def play_game(
    runner_policy: str = "random",
    chaser1_policy: str = "random", 
    chaser2_policy: str = "random",
    seed: Optional[int] = None,
    max_turns: int = 300
):
    """
    Play a single game with specified agent policies.
    
    Args:
        runner_policy: Policy for runner ("random" or "manual")
        chaser1_policy: Policy for chaser1 ("random" or "manual")
        chaser2_policy: Policy for chaser2 ("random" or "manual")
        seed: Random seed for reproducibility
        max_turns: Maximum number of turns
    """
    # Create board state
    board_state = BoardState(max_turns=max_turns, seed=seed)
    board_state.reset()
    
    # Create agents
    agents: Dict[str, BaseAgent] = {}
    
    for agent_name, policy in [
        ("runner", runner_policy),
        ("chaser1", chaser1_policy),
        ("chaser2", chaser2_policy)
    ]:
        if policy == "random":
            agents[agent_name] = RandomAgent(agent_name)
        elif policy == "manual":
            agents[agent_name] = ManualAgent(agent_name)
        else:
            raise ValueError(f"Unknown policy: {policy}")
    
    # Game loop
    print("\n=== CHASE GAME ===")
    print(f"Runner: {runner_policy}")
    print(f"Chaser1: {chaser1_policy}")
    print(f"Chaser2: {chaser2_policy}")
    if seed is not None:
        print(f"Seed: {seed}")
    print("\nStarting game...\n")
    
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
            time.sleep(0.1)
    
    # Game over - show final state
    print("\n" + "="*30)
    print(render_board(board_state, clear_screen=False))
    print(format_game_status(board_state))
    print("="*30)
    
    return board_state.winner


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Play the Chase Game in terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All random agents
  python cli_game.py
  
  # Manual runner vs random chasers
  python cli_game.py --runner manual
  
  # All manual agents
  python cli_game.py --runner manual --chaser1 manual --chaser2 manual
  
  # Set random seed for reproducibility
  python cli_game.py --seed 42
        """
    )
    
    parser.add_argument(
        "--runner",
        choices=["random", "manual"],
        default="random",
        help="Policy for the runner agent"
    )
    
    parser.add_argument(
        "--chaser1",
        choices=["random", "manual"],
        default="random",
        help="Policy for chaser1 agent"
    )
    
    parser.add_argument(
        "--chaser2",
        choices=["random", "manual"],
        default="random",
        help="Policy for chaser2 agent"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--max-turns",
        type=int,
        default=300,
        help="Maximum number of turns (default: 300)"
    )
    
    args = parser.parse_args()
    
    try:
        play_game(
            runner_policy=args.runner,
            chaser1_policy=args.chaser1,
            chaser2_policy=args.chaser2,
            seed=args.seed,
            max_turns=args.max_turns
        )
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()