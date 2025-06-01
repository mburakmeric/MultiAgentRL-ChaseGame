"""
Testing suite for the chase game environment.
Runs random rollouts and PettingZoo API compliance tests.
"""

import time
import numpy as np
from statistics import mean, median

from cgame_env import ChaseGameEnv
from pettingzoo.test import api_test


def run_random_rollout(env, render=False, max_steps=10000):
    """
    Run a single game with random actions.
    
    Args:
        env: The environment instance
        render: Whether to render the game
        max_steps: Maximum steps to prevent infinite loops
        
    Returns:
        Dictionary with game statistics
    """
    env.reset()
    
    steps = 0
    winner = None
    
    for agent in env.agent_iter(max_steps):
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            # Check winner
            if "win_condition" in info:
                if info["win_condition"] == "runner_win":
                    winner = "runner"
                elif info["win_condition"] == "chaser_win":
                    winner = "chasers"
            continue
            
        # Get action mask if available
        if "action_mask" in info:
            action_mask = info["action_mask"]
            legal_actions = np.where(action_mask)[0]
            action = np.random.choice(legal_actions)
        else:
            # Fallback to random action from action space
            action = env.action_space(agent).sample()
        
        env.step(action)
        steps += 1
        
        if render:
            env.render()
            time.sleep(0.1)
    
    # Calculate turns (3 steps = 1 turn)
    turns = steps // 3
    
    return {
        "steps": steps,
        "turns": turns,
        "winner": winner
    }


def run_multiple_games(n_games=100, seed=None, verbose=True):
    """
    Run multiple random games and collect statistics.
    
    Args:
        n_games: Number of games to run
        seed: Random seed for reproducibility
        verbose: Whether to print progress
        
    Returns:
        Dictionary with aggregated statistics
    """
    if seed is not None:
        np.random.seed(seed)
    
    stats = {
        "runner_wins": 0,
        "chaser_wins": 0,
        "turns_list": [],
        "steps_list": []
    }
    
    for i in range(n_games):
        if verbose and (i + 1) % 10 == 0:
            print(f"Running game {i + 1}/{n_games}...")
        
        env = ChaseGameEnv(seed=None if seed is None else seed + i)
        result = run_random_rollout(env, render=False)
        
        if result["winner"] == "runner":
            stats["runner_wins"] += 1
        elif result["winner"] == "chasers":
            stats["chaser_wins"] += 1
        
        stats["turns_list"].append(result["turns"])
        stats["steps_list"].append(result["steps"])
    
    # Calculate aggregated statistics
    stats["runner_win_rate"] = stats["runner_wins"] / n_games
    stats["chaser_win_rate"] = stats["chaser_wins"] / n_games
    stats["avg_turns"] = mean(stats["turns_list"])
    stats["median_turns"] = median(stats["turns_list"])
    stats["min_turns"] = min(stats["turns_list"])
    stats["max_turns"] = max(stats["turns_list"])
    
    return stats


def test_pettingzoo_api():
    """Test PettingZoo API compliance."""
    print("\n=== Testing PettingZoo API Compliance ===")
    
    env = ChaseGameEnv()
    
    try:
        api_test(env, num_cycles=10, verbose_progress=True)
        print("✓ PettingZoo API test passed!")
    except Exception as e:
        print(f"✗ PettingZoo API test failed: {e}")
        raise


def test_single_game_visual():
    """Run a single game with visualization."""
    print("\n=== Running Single Visual Test Game ===")
    
    env = ChaseGameEnv(seed=42, render_mode="human")
    
    result = run_random_rollout(env, render=True, max_steps=1000)
    
    print(f"\nGame finished!")
    print(f"Winner: {result['winner']}")
    print(f"Total turns: {result['turns']}")


def main():
    """Run all tests."""
    print("Chase Game Environment Tests")
    print("="*40)
    
    # Test 1: PettingZoo API compliance
    try:
        test_pettingzoo_api()
    except Exception as e:
        print(f"API test error: {e}")
    
    # Test 2: Run multiple random games
    print("\n=== Running Random Game Statistics ===")
    print("Running 100 random games...")
    
    stats = run_multiple_games(n_games=100, seed=42, verbose=True)
    
    print("\nResults:")
    print(f"Runner win rate: {stats['runner_win_rate']:.1%}")
    print(f"Chaser win rate: {stats['chaser_win_rate']:.1%}")
    print(f"Average game length: {stats['avg_turns']:.1f} turns")
    print(f"Median game length: {stats['median_turns']} turns")
    print(f"Shortest game: {stats['min_turns']} turns")
    print(f"Longest game: {stats['max_turns']} turns")
    
    # Test 3: Visual test (optional)
    user_input = input("\nRun visual test game? (y/N): ").strip().lower()
    if user_input == 'y':
        test_single_game_visual()
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    main()