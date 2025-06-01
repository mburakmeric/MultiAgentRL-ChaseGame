# RL Training Guide for Chase Game

This guide covers training reinforcement learning agents for the Chase Game using Stable-Baselines3 (SB3) and SuperSuit.

## Why Stable-Baselines3 + SuperSuit?

- **Stable-Baselines3**: Industry-standard RL library with clean API and reliable implementations
- **SuperSuit**: Provides wrappers to make PettingZoo environments compatible with SB3
- **Perfect for our needs**: Discrete actions, separate agent training, proven algorithms

## Installation

```bash
# Add to existing environment
pip install stable-baselines3
pip install supersuit

# Or install all RL dependencies
pip install stable-baselines3 supersuit tensorboard
```

## Training Architecture

Since each chaser must be autonomous (per project requirements), we'll train them separately:

1. **Train Chaser1**: Fix Runner and Chaser2 as random/pretrained, train Chaser1
2. **Train Chaser2**: Fix Runner and Chaser1 as random/pretrained, train Chaser2  
3. **Train Runner** (optional): Fix both chasers as trained/random, train Runner

This ensures each agent has its own neural network and makes decisions independently.

## Key Concepts

### Environment Wrapping
PettingZoo environments need to be wrapped for SB3 compatibility:
- Convert multi-agent env to single-agent perspective
- Vectorize for parallel training
- Handle action masking properly

### Algorithm Choice
**PPO (Proximal Policy Optimization)** - Recommended
- Works well with discrete actions
- Stable training
- Good for our sparse reward environment

**DQN (Deep Q-Network)** - Alternative
- Sample efficient
- Good for discrete action spaces
- May need reward shaping

### Training Considerations
- **Action Masking**: Already implemented in our environment
- **Observation Space**: Dict space with board state and positions
- **Reward Structure**: Sparse rewards (+1/-1 for win/loss)
- **Episode Length**: Max 300 turns (900 steps)

## Project Structure

```
version39/
├── training/
│   ├── train_chaser1.py      # Train first chaser
│   ├── train_chaser2.py      # Train second chaser
│   ├── train_runner.py       # Train runner (optional)
│   ├── evaluate_agents.py    # Test trained agents
│   └── models/               # Saved models directory
├── wrappers/
│   └── single_agent_wrapper.py  # Convert to single-agent perspective
└── configs/
    └── training_config.py    # Hyperparameters and settings
```

## Next Steps

1. Create wrapper to convert PettingZoo env to single-agent perspective
2. Implement training scripts for each agent
3. Set up hyperparameter configurations
4. Create evaluation scripts to test trained agents
5. Implement tournament system to compare different strategies

## Useful Resources

- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [SuperSuit Docs](https://github.com/Farama-Foundation/SuperSuit)
- [PettingZoo Training Examples](https://pettingzoo.farama.org/tutorials/sb3/connect_four/)

## Notes

- Each chaser will have a separate model file (e.g., `chaser1_ppo.zip`, `chaser2_ppo.zip`)
- Models can be loaded and used independently during gameplay
- Consider curriculum learning: start with easier scenarios and increase difficulty