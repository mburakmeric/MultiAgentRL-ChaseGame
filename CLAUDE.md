# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a grid-based multi-agent chase game implementing a PettingZoo environment. The game features three agents (Runner, Chaser1, Chaser2) on a 15x15 grid with obstacles, designed to be modular, terminal-playable, and RL-ready.

## Architecture

The project follows a modular structure with clear separation of concerns:

- **cgame_env.py**: PettingZoo environment wrapper connecting board logic to RL APIs
- **board_state.py**: Core game logic (grid state, movement, rules, win conditions)
- **agent_policies.py**: Agent behavior interfaces (BaseAgent, RandomAgent, ManualAgent)
- **cli_game.py**: Terminal game runner for interactive play
- **utils.py**: Helper functions (board rendering, input helpers, factories)
- **test_env.py**: Testing suite for random rollouts and PettingZoo compliance

## Key Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pettingzoo numpy
# Optional: pip install rich typer
```

### Running the Game
```bash
# Run CLI game (once implemented)
python cli_game.py

# Run tests
python test_env.py
```

## Game Rules & Constraints

### Board Specifications
- Fixed 15x15 grid (rows 1-15, columns a-o)
- 5 obstacles per game
- 300 turn limit (Runner wins if not caught)

### Turn Order & Counting
- **Order**: Runner → Chaser1 → Chaser2 (always this sequence)
- **Turn counting**: One complete round (all 3 agents move) = 1 turn toward 300-turn limit
- **Win check timing**: After EACH individual agent moves (not just at round end)

### Agent Movement
- Actions: Up, Down, Left, Right, Stay
- Stay only allowed when ALL four directional moves blocked (invalid otherwise - agent must choose valid move)
- Movement into occupied cells:
  - Chasers can move into each other's cell (overlap allowed)
  - Any agent moving into Runner's cell = capture (game ends)
  - Runner moving into Chaser's cell = capture (game ends)

### Spawning Rules
1. Agents placed first with Manhattan distance ≥ 7 between Runner and each Chaser
2. Obstacles forbidden within Chebyshev distance ≤ 1 of any agent spawn
3. All placement must be reproducible with fixed random seed

### Win Conditions
- **Chasers win**: Runner caught (shares cell with chaser) - checked after each agent's move
- **Runner wins**: Survives 300 complete turns OR completely surrounded by obstacles/edges (no chasers present)

## Implementation Guidelines

### Board State Representation
- Use numpy array (15x15) with integer encoding:
  - 0: empty
  - 1: obstacle
  - 2: runner
  - 3: chaser1
  - 4: chaser2
- Track chaser positions separately; handle overlap during rendering/logic checks
- When both chasers overlap, keep their individual positions (don't use special encoding)

### PettingZoo Integration
- Subclass `pettingzoo.AECEnv`
- Fixed agent order: ["runner", "chaser1", "chaser2"]
- Action space: Discrete(5) - [up, down, left, right, stay]
- Observation space: Dictionary with:
  ```python
  {
    "board": 15x15 numpy array,
    "runner_pos": (x, y),
    "chaser1_pos": (x, y),
    "chaser2_pos": (x, y)
  }
  ```
- Action masking: Implement to prevent invalid actions (including invalid Stay)
- Rewards:
  - Runner: +1 for winning, -1 if caught, 0 otherwise
  - Chasers: +1 for catching runner, 0 otherwise
- Info dict should include:
  - `turn_number`: Current turn count
  - `legal_moves`: List of valid actions for current agent
  - `win_condition`: "runner_win", "chaser_win", or None

### ASCII Rendering
- █ = obstacle
- R = runner
- C1/C2 = individual chasers
- C = overlapping chasers
- . = empty cell

## Testing Approach

Use `pettingzoo.test.api_test` to validate environment compliance. Run multiple random games to verify:
- Proper spawning constraints
- Win condition logic
- Move validation
- Reproducibility with seeds

## Important Notes

- No file/network I/O beyond CLI in initial version
- CLI features will be expanded in later phases
- All random elements must support fixed seeds for reproducibility
- Focus on modularity to allow easy agent policy swapping