"""
GPU Stress Test Script for Chase Game Environment
Tests different configurations to maximize GPU utilization
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import torch
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from gymnasium import spaces
import torch.nn as nn

from wrappers.single_agent_wrapper import make_training_env


class LargeCustomCNN(BaseFeaturesExtractor):
    """
    Large CNN feature extractor to stress GPU
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        board_shape = observation_space["board"].shape
        n_input_channels = board_shape[0] if len(board_shape) == 3 else 1
        
        # Much larger CNN than needed - for GPU stress testing
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            sample_board = torch.zeros(1, n_input_channels, *board_shape[-2:])
            cnn_out_size = self.cnn(sample_board).shape[1]
        
        # Position features
        position_size = 6  # 3 positions * 2 coordinates
        
        # Large fully connected layers
        self.linear = nn.Sequential(
            nn.Linear(cnn_out_size + position_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, features_dim),
            nn.ReLU(),
        )
        
    def forward(self, observations):
        # Extract board and positions
        board = observations["board"].float()
        
        # Add channel dimension if needed
        if len(board.shape) == 3:
            board = board.unsqueeze(1)
            
        # CNN processing
        cnn_out = self.cnn(board)
        
        # Concatenate positions
        positions = torch.cat([
            observations["runner_pos"].float(),
            observations["chaser1_pos"].float(),
            observations["chaser2_pos"].float()
        ], dim=1)
        
        # Combine all features
        combined = torch.cat([cnn_out, positions], dim=1)
        
        return self.linear(combined)


def create_stress_test_policy():
    """Create a policy with large networks for GPU stress testing"""
    policy_kwargs = dict(
        features_extractor_class=LargeCustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(
            pi=[1024, 1024, 512, 512],  # Large policy network
            vf=[1024, 1024, 512, 512]   # Large value network
        ),
        activation_fn=nn.ReLU
    )
    return policy_kwargs


def run_gpu_stress_test(test_name, n_envs, batch_size, n_steps, total_timesteps=100000):
    """Run a single stress test configuration"""
    print(f"\n{'='*60}")
    print(f"Running Test: {test_name}")
    print(f"Environments: {n_envs}, Batch Size: {batch_size}, Steps: {n_steps}")
    print(f"{'='*60}")
    
    # Create environments
    def make_env_lambda(rank):
        return lambda: make_training_env("chaser1", opponent_policy_types={"runner": "random", "chaser2": "random"}, seed=rank)
    
    env = SubprocVecEnv([make_env_lambda(i) for i in range(n_envs)])
    
    # Create model with large networks
    model = PPO(
        "MultiInputPolicy",
        env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=20,  # More epochs for more GPU work
        learning_rate=1e-4,
        policy_kwargs=create_stress_test_policy(),
        verbose=1,
        device="cuda"  # Force CUDA
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Total model parameters: {total_params:,}")
    print(f"Model device: {model.device}")
    
    # Monitor GPU before training
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        print(f"GPU memory allocated: {start_memory:.2f} GB")
    
    # Time the training
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True
        )
    except Exception as e:
        print(f"Test failed with error: {e}")
        
    end_time = time.time()
    
    # Report results
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        end_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        max_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        print(f"\nGPU Memory Usage:")
        print(f"  Final: {end_memory:.2f} GB")
        print(f"  Peak: {max_memory:.2f} GB")
        
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print(f"Speed: {total_timesteps / (end_time - start_time):.0f} steps/second")
    
    # Cleanup
    env.close()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return end_time - start_time


def run_image_based_stress_test():
    """Test with image observations for maximum GPU stress"""
    print(f"\n{'='*60}")
    print("Running Image-Based Stress Test")
    print("Creating fake image observations (84x84x3)")
    print(f"{'='*60}")
    
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    import gymnasium as gym
    
    class FakeImageEnv(gym.Env):
        """Fake environment with image observations"""
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8)
            self.action_space = spaces.Discrete(5)
            self.step_count = 0
            
        def reset(self, seed=None, options=None):
            self.step_count = 0
            return np.random.randint(0, 255, size=(84, 84, 3), dtype=np.uint8), {}
            
        def step(self, action):
            self.step_count += 1
            obs = np.random.randint(0, 255, size=(84, 84, 3), dtype=np.uint8)
            done = self.step_count >= 100
            return obs, 0.0, done, False, {}
        
        def render(self):
            pass
            
        def close(self):
            pass
    
    # Create environments
    env = DummyVecEnv([lambda: FakeImageEnv() for _ in range(32)])
    
    # CNN policy for images
    model = PPO(
        "CnnPolicy",
        env,
        n_steps=128,
        batch_size=256,
        n_epochs=10,
        learning_rate=1e-4,
        verbose=1,
        device="cuda"
    )
    
    print(f"Total parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    start_time = time.time()
    model.learn(total_timesteps=50000, progress_bar=True)
    end_time = time.time()
    
    print(f"Image-based training time: {end_time - start_time:.2f} seconds")
    print(f"Speed: {50000 / (end_time - start_time):.0f} steps/second")
    
    env.close()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    """Run various GPU stress tests"""
    
    # Check CUDA availability
    print("="*60)
    print("GPU STRESS TEST FOR CHASE GAME")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("CUDA Available: No - Tests will run on CPU")
    
    # Test configurations (adjusted for GTX 970 4GB)
    tests = [
        # (name, n_envs, batch_size, n_steps)
        ("Small Test", 4, 64, 128),
        ("Medium Test", 8, 128, 256),
        ("Large Test", 16, 256, 256),
        ("XL Test", 24, 256, 128),
        ("Max Batch Size", 8, 512, 256),
    ]
    
    results = {}
    
    # Run stress tests
    for test_config in tests:
        try:
            time_taken = run_gpu_stress_test(*test_config)
            results[test_config[0]] = time_taken
        except Exception as e:
            print(f"\nTest '{test_config[0]}' failed: {e}")
            results[test_config[0]] = "Failed"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Run image-based test
    try:
        run_image_based_stress_test()
    except Exception as e:
        print(f"\nImage test failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("STRESS TEST SUMMARY")
    print("="*60)
    for test_name, result in results.items():
        if isinstance(result, float):
            print(f"{test_name}: {result:.2f} seconds")
        else:
            print(f"{test_name}: {result}")
    
    # GPU memory summary
    if torch.cuda.is_available():
        print(f"\nFinal GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    print("\nNote: For maximum GPU usage, try:")
    print("- Increase batch_size and n_epochs in training_config.py")
    print("- Use more complex neural networks")
    print("- Process image observations instead of simple arrays")
    print("- Run multiple training scripts in parallel")


if __name__ == "__main__":
    main()