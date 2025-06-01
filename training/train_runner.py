"""
Training script for Runner agent using Stable-Baselines3.
Can optionally load trained Chaser models as opponents.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime
import numpy as np

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from wrappers.single_agent_wrapper import make_training_env, SingleAgentWrapper
from configs.training_config import get_agent_config
from agent_policies import BaseAgent, RandomAgent


class SB3PolicyAgent(BaseAgent):
    """Wrapper to use a trained SB3 model as an agent policy."""
    
    def __init__(self, name: str, model_path: str):
        super().__init__(name)
        # Load the trained model
        if "ppo" in model_path.lower():
            self.model = PPO.load(model_path)
        elif "dqn" in model_path.lower():
            self.model = DQN.load(model_path)
        else:
            # Try PPO by default
            self.model = PPO.load(model_path)
    
    def decide_move(self, observation, legal_moves):
        """Use the trained model to decide the move."""
        # Model expects observation in the right format
        action, _ = self.model.predict(observation, deterministic=True)
        
        # Ensure the action is legal
        if action in legal_moves:
            return action
        else:
            # Fallback to random legal move if model suggests illegal action
            return np.random.choice(legal_moves)


def make_env(agent_name: str, rank: int, seed: int = 0, 
             chaser1_model_path: str = None, chaser2_model_path: str = None):
    """
    Create a wrapped environment for training.
    
    Args:
        agent_name: Name of agent to train
        rank: Unique ID for the environment
        seed: Random seed
        chaser1_model_path: Path to trained Chaser1 model (optional)
        chaser2_model_path: Path to trained Chaser2 model (optional)
    """
    def _init():
        config = get_agent_config(agent_name)
        opponent_policies = {}
        
        # Set up opponent policies
        for opp_name, policy_type in config["opponent_policies"].items():
            if opp_name == "chaser1" and chaser1_model_path:
                # Use trained Chaser1 model
                opponent_policies[opp_name] = SB3PolicyAgent(opp_name, chaser1_model_path)
            elif opp_name == "chaser2" and chaser2_model_path:
                # Use trained Chaser2 model
                opponent_policies[opp_name] = SB3PolicyAgent(opp_name, chaser2_model_path)
            elif policy_type == "random":
                opponent_policies[opp_name] = RandomAgent(opp_name)
        
        env = SingleAgentWrapper(
            agent_to_train=agent_name,
            opponent_policies=opponent_policies,
            seed=seed + rank
        )
        env = Monitor(env)
        return env
    
    set_random_seed(seed)
    return _init


def train_runner(args):
    """Train Runner agent."""
    agent_name = "runner"
    config = get_agent_config(agent_name)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up directories
    model_dir = f"./training/models/{agent_name}_{timestamp}"
    log_dir = f"./training/logs/{agent_name}_{timestamp}"
    tensorboard_dir = f"./training/tensorboard/{agent_name}_{timestamp}"
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Create training environments
    if args.n_envs > 1:
        env = SubprocVecEnv([
            make_env(agent_name, i, args.seed, args.chaser1_model, args.chaser2_model) 
            for i in range(args.n_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env(agent_name, 0, args.seed, args.chaser1_model, args.chaser2_model)
        ])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([
        make_env(agent_name, 99, args.seed, args.chaser1_model, args.chaser2_model)
    ])
    
    # Select algorithm
    algorithm_config = config["algorithm_config"].copy()
    algorithm_config["tensorboard_log"] = tensorboard_dir
    
    if config["algorithm"] == "PPO":
        model = PPO(
            "MultiInputPolicy",
            env,
            **algorithm_config
        )
    elif config["algorithm"] == "DQN":
        model = DQN(
            "MultiInputPolicy",
            env,
            **algorithm_config
        )
    else:
        raise ValueError(f"Unknown algorithm: {config['algorithm']}")
    
    # Set up callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/best",
        log_path=log_dir,
        eval_freq=config["training_config"]["eval_freq"],
        n_eval_episodes=config["training_config"]["n_eval_episodes"],
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config["training_config"]["save_freq"],
        save_path=f"{model_dir}/checkpoints",
        name_prefix=f"{agent_name}_checkpoint"
    )
    
    callbacks = CallbackList([eval_callback, checkpoint_callback])
    
    # Train the model
    print(f"\nStarting training for {agent_name}")
    print(f"Algorithm: {config['algorithm']}")
    print(f"Total timesteps: {args.timesteps}")
    print(f"Number of environments: {args.n_envs}")
    
    if args.chaser1_model:
        print(f"Using trained Chaser1 model: {args.chaser1_model}")
    else:
        print("Using random Chaser1 opponent")
        
    if args.chaser2_model:
        print(f"Using trained Chaser2 model: {args.chaser2_model}")
    else:
        print("Using random Chaser2 opponent")
        
    print(f"Models will be saved to: {model_dir}")
    print(f"Tensorboard logs: {tensorboard_dir}")
    print("\nTo monitor training, run:")
    print(f"tensorboard --logdir {tensorboard_dir}")
    print("-" * 50)
    
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = f"{model_dir}/final_model"
    model.save(final_model_path)
    print(f"\nTraining completed! Final model saved to: {final_model_path}")
    
    # Test the trained model
    if args.test_episodes > 0:
        print(f"\nTesting trained model for {args.test_episodes} episodes...")
        
        total_rewards = []
        wins = 0
        
        for episode in range(args.test_episodes):
            print(f"Running test episode {episode + 1}/{args.test_episodes}...", end='\r')
            obs = eval_env.reset()
            episode_reward = 0
            done = False
            steps = 0
            max_steps = 1000  # Safety limit
            
            while steps < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, info = eval_env.step(action)
                episode_reward += reward[0]
                steps += 1
                
                # Handle both single and vectorized done signals
                if isinstance(dones, np.ndarray):
                    done = dones[0]
                else:
                    done = dones
                    
                if done:
                    break
                
            total_rewards.append(episode_reward)
            if episode_reward > 0:
                wins += 1
            
        print(f"\nTest Results:")
        print(f"Win rate: {wins}/{args.test_episodes} ({100*wins/args.test_episodes:.1f}%)")
        print(f"Average reward: {np.mean(total_rewards):.3f}")
        print(f"Std reward: {np.std(total_rewards):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train Runner agent")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=10,
        help="Number of test episodes after training (0 to skip)"
    )
    parser.add_argument(
        "--chaser1-model",
        type=str,
        default=None,
        help="Path to trained Chaser1 model (optional)"
    )
    parser.add_argument(
        "--chaser2-model",
        type=str,
        default=None,
        help="Path to trained Chaser2 model (optional)"
    )
    
    args = parser.parse_args()
    train_runner(args)


if __name__ == "__main__":
    main()