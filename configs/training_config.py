"""
Configuration settings for RL training.
"""

from typing import Dict, Any


# PPO hyperparameters
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 512,  # Reduced for more frequent updates
    "batch_size": 128,  # Increased batch size
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1,
    "tensorboard_log": "./training/tensorboard/"
}

# DQN hyperparameters (alternative)
DQN_CONFIG = {
    "learning_rate": 1e-4,
    "buffer_size": 50000,
    "learning_starts": 1000,
    "batch_size": 32,
    "tau": 1.0,
    "gamma": 0.99,
    "train_freq": 16,  # Less frequent training = faster
    "gradient_steps": 1,
    "target_update_interval": 5000,  # Less frequent target updates
    "exploration_fraction": 0.1,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "verbose": 1,
    "tensorboard_log": "./training/tensorboard/"
}

# Training settings
TRAINING_CONFIG = {
    "total_timesteps": 500000,  # Total training steps
    "eval_freq": 10000,         # Evaluate every N steps
    "n_eval_episodes": 10,      # Number of episodes for evaluation
    "save_freq": 50000,         # Save model every N steps
    "log_interval": 100,        # Log stats every N episodes
}

# Environment settings
ENV_CONFIG = {
    "board_size": 15,
    "max_turns": 300,
    "n_envs": 8,  # Number of parallel environments for training
}

# Agent-specific configurations
AGENT_CONFIGS = {
    "chaser1": {
        "algorithm": "PPO",  # PPO for chasers
        "model_name": "chaser1_ppo",
        "opponent_policies": {
            "runner": "random",
            "chaser2": "random"
        }
    },
    "chaser2": {
        "algorithm": "PPO",  # PPO for chasers
        "model_name": "chaser2_ppo",
        "opponent_policies": {
            "runner": "random",
            "chaser1": "random"  # Can load trained chaser1 later
        }
    },
    "runner": {
        "algorithm": "DQN",  # DQN for runner
        "model_name": "runner_dqn",
        "opponent_policies": {
            "chaser1": "random",  # Can load trained chasers later
            "chaser2": "random"
        }
    }
}


def get_algorithm_config(algorithm: str) -> Dict[str, Any]:
    """Get hyperparameters for specified algorithm."""
    if algorithm == "PPO":
        return PPO_CONFIG
    elif algorithm == "DQN":
        return DQN_CONFIG
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def get_agent_config(agent_name: str) -> Dict[str, Any]:
    """Get configuration for specific agent."""
    if agent_name not in AGENT_CONFIGS:
        raise ValueError(f"Unknown agent: {agent_name}")
    
    config = AGENT_CONFIGS[agent_name].copy()
    config["algorithm_config"] = get_algorithm_config(config["algorithm"])
    config["training_config"] = TRAINING_CONFIG.copy()
    config["env_config"] = ENV_CONFIG.copy()
    
    return config