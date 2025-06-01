"""
Evaluation script to test trained agents against each other.
Supports various agent combinations and visualizations.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from typing import Dict, Optional

from stable_baselines3 import PPO, DQN

from cgame_env import ChaseGameEnv
from agent_policies import BaseAgent, RandomAgent, ManualAgent
from utils import render_board, format_game_status


class SB3PolicyAgent(BaseAgent):
    """Wrapper to use a trained SB3 model as an agent policy."""
    
    def __init__(self, name: str, model_path: str):
        super().__init__(name)
        print(f"Loading model for {name} from: {model_path}")
        
        # Load the trained model
        if "ppo" in model_path.lower():
            self.model = PPO.load(model_path)
        elif "dqn" in model_path.lower():
            self.model = DQN.load(model_path)
        else:
            # Try PPO by default
            try:
                self.model = PPO.load(model_path)
            except:
                self.model = DQN.load(model_path)
    
    def decide_move(self, observation, legal_moves):
        """Use the trained model to decide the move."""
        # Model expects observation in the right format
        action, _ = self.model.predict(observation, deterministic=True)
        
        # Ensure the action is legal
        if action in legal_moves:
            return action
        else:
            # Fallback to random legal move if model suggests illegal action
            print(f"Warning: {self.name} suggested illegal action {action}, choosing random from {legal_moves}")
            return np.random.choice(legal_moves)


def load_agent(agent_name: str, agent_type: str, model_path: Optional[str] = None) -> BaseAgent:
    """Load an agent based on type specification."""
    if agent_type == "random":
        return RandomAgent(agent_name)
    elif agent_type == "manual":
        return ManualAgent(agent_name)
    elif agent_type == "trained":
        if not model_path:
            raise ValueError(f"Model path required for trained agent {agent_name}")
        return SB3PolicyAgent(agent_name, model_path)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def run_evaluation(
    runner_type: str,
    chaser1_type: str, 
    chaser2_type: str,
    runner_model: Optional[str] = None,
    chaser1_model: Optional[str] = None,
    chaser2_model: Optional[str] = None,
    n_episodes: int = 10,
    seed: Optional[int] = None,
    render: bool = False,
    render_delay: float = 0.1
):
    """
    Run evaluation games with specified agent configurations.
    
    Returns:
        Dictionary with evaluation statistics
    """
    # Create environment
    env = ChaseGameEnv(seed=seed, render_mode="human" if render else None)
    
    # Load agents
    agents = {
        "runner": load_agent("runner", runner_type, runner_model),
        "chaser1": load_agent("chaser1", chaser1_type, chaser1_model),
        "chaser2": load_agent("chaser2", chaser2_type, chaser2_model)
    }
    
    # Statistics tracking
    stats = {
        "runner_wins": 0,
        "chaser_wins": 0,
        "turn_counts": [],
        "win_details": []
    }
    
    print(f"\nStarting evaluation for {n_episodes} episodes")
    print(f"Runner: {runner_type}" + (f" ({runner_model})" if runner_model else ""))
    print(f"Chaser1: {chaser1_type}" + (f" ({chaser1_model})" if chaser1_model else ""))
    print(f"Chaser2: {chaser2_type}" + (f" ({chaser2_model})" if chaser2_model else ""))
    print("-" * 50)
    
    for episode in range(n_episodes):
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        env.reset()
        
        # Run episode
        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                continue
            
            # Get agent's action
            agent = agents[agent_name]
            legal_moves = info.get("legal_moves", list(range(5)))
            action = agent.decide_move(observation, legal_moves)
            
            env.step(action)
            
            if render:
                env.render()
                import time
                time.sleep(render_delay)
        
        # Collect episode results
        final_info = env.unwrap().infos[env.agents[0]]
        win_condition = final_info.get("win_condition", None)
        turn_number = final_info.get("turn_number", 0)
        
        if win_condition == "runner_win":
            stats["runner_wins"] += 1
            winner = "Runner"
        elif win_condition == "chaser_win":
            stats["chaser_wins"] += 1
            winner = "Chasers"
        else:
            winner = "Unknown"
        
        stats["turn_counts"].append(turn_number)
        stats["win_details"].append({
            "episode": episode + 1,
            "winner": winner,
            "turns": turn_number,
            "win_condition": win_condition
        })
        
        print(f"Episode {episode + 1} result: {winner} wins after {turn_number} turns")
    
    # Calculate summary statistics
    stats["runner_win_rate"] = stats["runner_wins"] / n_episodes
    stats["chaser_win_rate"] = stats["chaser_wins"] / n_episodes
    stats["avg_turns"] = np.mean(stats["turn_counts"])
    stats["std_turns"] = np.std(stats["turn_counts"])
    stats["min_turns"] = min(stats["turn_counts"])
    stats["max_turns"] = max(stats["turn_counts"])
    
    return stats


def print_evaluation_summary(stats: Dict):
    """Print a nice summary of evaluation results."""
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total episodes: {len(stats['win_details'])}")
    print(f"\nWin rates:")
    print(f"  Runner:  {stats['runner_wins']}/{len(stats['win_details'])} ({stats['runner_win_rate']:.1%})")
    print(f"  Chasers: {stats['chaser_wins']}/{len(stats['win_details'])} ({stats['chaser_win_rate']:.1%})")
    print(f"\nGame length statistics:")
    print(f"  Average: {stats['avg_turns']:.1f} turns")
    print(f"  Std Dev: {stats['std_turns']:.1f} turns")
    print(f"  Min:     {stats['min_turns']} turns")
    print(f"  Max:     {stats['max_turns']} turns")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agents")
    
    # Agent type arguments
    parser.add_argument(
        "--runner",
        choices=["random", "manual", "trained"],
        default="random",
        help="Type of runner agent"
    )
    parser.add_argument(
        "--chaser1",
        choices=["random", "manual", "trained"],
        default="random",
        help="Type of chaser1 agent"
    )
    parser.add_argument(
        "--chaser2",
        choices=["random", "manual", "trained"],
        default="random",
        help="Type of chaser2 agent"
    )
    
    # Model path arguments
    parser.add_argument(
        "--runner-model",
        type=str,
        help="Path to trained runner model"
    )
    parser.add_argument(
        "--chaser1-model",
        type=str,
        help="Path to trained chaser1 model"
    )
    parser.add_argument(
        "--chaser2-model",
        type=str,
        help="Path to trained chaser2 model"
    )
    
    # Evaluation settings
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render games visually"
    )
    parser.add_argument(
        "--render-delay",
        type=float,
        default=0.1,
        help="Delay between renders in seconds"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.runner == "trained" and not args.runner_model:
        parser.error("--runner-model required when --runner is 'trained'")
    if args.chaser1 == "trained" and not args.chaser1_model:
        parser.error("--chaser1-model required when --chaser1 is 'trained'")
    if args.chaser2 == "trained" and not args.chaser2_model:
        parser.error("--chaser2-model required when --chaser2 is 'trained'")
    
    # Run evaluation
    stats = run_evaluation(
        runner_type=args.runner,
        chaser1_type=args.chaser1,
        chaser2_type=args.chaser2,
        runner_model=args.runner_model,
        chaser1_model=args.chaser1_model,
        chaser2_model=args.chaser2_model,
        n_episodes=args.episodes,
        seed=args.seed,
        render=args.render,
        render_delay=args.render_delay
    )
    
    # Print summary
    print_evaluation_summary(stats)


if __name__ == "__main__":
    main()