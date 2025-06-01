# Chase Game - Grid-Based Multi-Agent Environment

A modular, terminal-playable, and RL-ready grid-based chase game implemented with PettingZoo. Three agents (Runner, Chaser1, Chaser2) compete on a 15x15 grid with obstacles.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd version40

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pettingzoo numpy

# For RL training (optional)
pip install stable-baselines3 supersuit tensorboard
```

## Quick Start

### Play in Terminal

#### Standard CLI (Command-line arguments)
```bash
# All agents play randomly
python cli_game.py

# Manual control for runner
python cli_game.py --runner manual

# All manual control
python cli_game.py --runner manual --chaser1 manual --chaser2 manual

# Set random seed for reproducible games
python cli_game.py --seed 42
```

#### Interactive CLI (Prompt-based setup)
```bash
# Interactive setup for demos and custom games
python cli_game_interactive.py
```

The interactive CLI provides a complete setup experience:
- Custom number of turns and obstacles
- Choose random or manual obstacle placement
- Manually specify agent starting positions (format: "a 3", "h 8", etc.)
- Select agent control methods at runtime

### Run Tests

```bash
# Run test suite (API compliance + statistics)
python test_env.py
```

### Train RL Agents

```bash
# Train individual agents (each chaser is autonomous)
python training/train_chaser1.py
python training/train_chaser2.py
python training/train_runner.py

# Evaluate trained agents
python training/evaluate_agents.py
```

See `RL_TRAINING.md` for detailed training instructions and hyperparameter configuration.

## Game Rules

- **Board**: 15x15 grid with 5 randomly placed obstacles
- **Agents**: Runner (R), Chaser1 (C1), Chaser2 (C2)
- **Turn Order**: Runner → Chaser1 → Chaser2 (repeating)
- **Actions**: Up, Down, Left, Right, Stay (only when blocked)
- **Win Conditions**:
  - Chasers win if Runner is caught (shares cell with any chaser)
  - Runner wins if survives 300 turns OR is completely surrounded by obstacles/edges

## Using as PettingZoo Environment

```python
from cgame_env import ChaseGameEnv

# Create environment
env = ChaseGameEnv(seed=42)
env.reset()

# Run with random actions
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    
    if termination or truncation:
        action = None
    else:
        # Use action mask for valid actions
        action_mask = info.get("action_mask")
        if action_mask is not None:
            legal_actions = np.where(action_mask)[0]
            action = np.random.choice(legal_actions)
        else:
            action = env.action_space(agent).sample()
    
    env.step(action)
```

## Swapping Agent Policies

The modular design allows easy swapping of agent behaviors:

```python
from board_state import BoardState
from agent_policies import RandomAgent, ManualAgent, BaseAgent

# Create custom agent
class SmartChaser(BaseAgent):
    def decide_move(self, observation, legal_moves):
        # Your custom logic here
        board = observation["board"]
        runner_pos = observation["runner_pos"]
        my_pos = observation[f"{self.name}_pos"]
        
        # Example: move towards runner
        # ... implement chase logic ...
        
        return chosen_action

# Use in CLI game
# Modify cli_game.py to import and use your custom agent
```

## Project Structure

```
version40/
├── board_state.py            # Core game logic
├── agent_policies.py         # Agent behavior interfaces
├── cgame_env.py             # PettingZoo environment wrapper
├── cli_game.py              # Standard terminal game runner
├── cli_game_interactive.py  # Interactive prompt-based CLI
├── utils.py                 # Rendering and helper functions
├── test_env.py              # Test suite
├── configs/
│   └── training_config.py   # RL training hyperparameters
├── training/
│   ├── train_chaser1.py     # Train first chaser
│   ├── train_chaser2.py     # Train second chaser
│   ├── train_runner.py      # Train runner (optional)
│   ├── evaluate_agents.py   # Test trained agents
│   └── models/              # Saved models directory
├── wrappers/
│   └── single_agent_wrapper.py  # Multi-agent to single-agent conversion
├── README.md                # This file
├── CLAUDE.md               # Development guidance
└── RL_TRAINING.md          # RL training guide
```

## Environment Details

### Observation Space
```python
{
    "board": 15x15 numpy array (0=empty, 1=obstacle, 2=runner, 3=chaser1, 4=chaser2),
    "runner_pos": (x, y),
    "chaser1_pos": (x, y),
    "chaser2_pos": (x, y)
}
```

### Action Space
- Discrete(5): [0=Up, 1=Down, 2=Left, 3=Right, 4=Stay]

### Rewards
- Runner: +1 for winning, -1 if caught, 0 otherwise
- Chasers: +1 for catching runner, 0 otherwise

### Info Dictionary
```python
{
    "turn_number": int,
    "legal_moves": list of valid actions,
    "win_condition": "runner_win" | "chaser_win" | None,
    "action_mask": binary array of legal actions
}
```

## Manual Controls

When playing with `--runner manual` or `--chaser manual`:
- **U**: Move up
- **D**: Move down
- **L**: Move left
- **R**: Move right
- **S**: Stay (only when no other moves available)

## Development

To extend or modify the game:

1. **New Agent Types**: Inherit from `BaseAgent` in `agent_policies.py`
2. **Board Modifications**: Edit `BoardState` in `board_state.py`
3. **Rendering Changes**: Modify functions in `utils.py`
4. **RL Training**: Use the `ChaseGameEnv` with standard RL libraries

## Reproducibility

All random elements (agent spawning, obstacle placement) support fixed seeds:

```bash
# Command line
python cli_game.py --seed 123

# In code
env = ChaseGameEnv(seed=123)
```

## Current Status

- ✅ Core game engine complete
- ✅ PettingZoo environment implementation
- ✅ Two CLI interfaces (standard and interactive)
- ✅ RL training pipeline with Stable-Baselines3
- ✅ Autonomous agent architecture (each chaser independent)
- 🔄 Neural network integration into CLI (future work)
- 🔄 Tournament system for agent comparison (future work)

## Troubleshooting

If you encounter training errors:
- Ensure all dependencies are installed: `pip install stable-baselines3 supersuit tensorboard`
- Check Python version compatibility (3.8+ recommended)
- Verify CUDA setup if using GPU acceleration
- See training logs in `training/logs/` for detailed error messages

## License

[Your license here]