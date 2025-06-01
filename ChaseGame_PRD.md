
# Grid-Based Multi-Agent Chase Game — PRD v2.3
*(PettingZoo + Modular Structure + Detailed Game World, with CLI Development Note)*

---

## 1 · Scope & Purpose

Design a **modular, terminal-playable, and RL-ready** grid-based chase game, where all three agents initially move randomly but can be replaced with learned or rule-based agents with no changes to core game logic.  
The project should be easy to test, debug, and extend.

---

## 2 · Architecture & File Layout

**Keep all major responsibilities in separate files for clarity and modularity.**

```
cgame_folder/
├── cgame_env.py        # PettingZoo environment: connects board logic to RL APIs
├── board_state.py      # Core board logic: grid, spawning, movement, rules
├── agent_policies.py   # BaseAgent + all policy logic (random, manual, RL hook)
├── cli_game.py         # Terminal game runner: handles user setup and play (initial version; details TBD)
├── utils.py            # Board rendering (ASCII), input helpers, factories
├── test_env.py         # Simple random rollout tests and PettingZoo env checks
└── README.md           # Setup, usage, and policy swap instructions
```

---

## 3 · Game World (Detailed)

### **3.1 Board Dimensions**
- The board is a **fixed 15x15 grid**.
    - **Rows:** Numbered `1` (top) to `15` (bottom).
    - **Columns:** Labeled `a` (left) to `o` (right).
    - Each cell can be addressed as `"c7"` (column `c`, row `7`).

---

### **3.2 Agents**
- There are **three agents:**
    - **Runner** (`R`)
    - **Chaser1** (`C1`)
    - **Chaser2** (`C2`)
- All agent names must be unique and fixed: `"runner"`, `"chaser1"`, `"chaser2"`.
- Each agent **occupies one cell** at a time.

---

### **3.3 Obstacles ("Black Cells")**
- There are **exactly 5 obstacles** on the board each game.
    - Obstacles are displayed as the `█` symbol in the ASCII renderer.
    - Obstacles cannot be moved or removed during a game.

---

### **3.4 Spawning and Placement Rules**

#### **Step 1: Place Agents**
- All three agents must be placed on **distinct, empty cells**.
- Agent positions must be chosen so that:
    - The **Manhattan distance** (|x1–x2| + |y1–y2|) between the Runner and each Chaser (R–C1 and R–C2) is **at least 7**.
    - Chasers may be placed closer to each other, but not on the same square as the Runner.
- Agent spawn positions must be chosen *before* obstacle placement.

#### **Step 2: Compute Forbidden Zones for Obstacles**
- For each agent’s starting cell, **forbid obstacle placement** in the **3x3 square** centered on that cell.
    - This is defined as all cells within **Chebyshev distance ≤ 1** of each agent's spawn.
        - Chebyshev distance: max(|x1–x2|, |y1–y2|).
    - This prevents obstacles from boxing in any agent at game start.

#### **Step 3: Place Obstacles**
- From all legal (not forbidden) empty cells, select **5 at random** to become obstacles.
- **Obstacles must never** overlap an agent’s starting cell or forbidden zones.
- No further pathfinding or flood-fill checks are needed; this is sufficient to guarantee all agents have at least one open path at start.

---

### **3.5 Cell State Encoding**
- Each cell is always in exactly one state:
    - **Empty**
    - **Obstacle**
    - **Runner**
    - **Chaser1**
    - **Chaser2**
    - (If both Chasers occupy the same cell, display as a single `C` in the ASCII renderer.)

---

### **3.6 Board Representation**
- Internally, represent the board as a **15×15 numpy array**.
    - Example encoding:
        - `0`: empty cell
        - `1`: obstacle
        - `2`: runner
        - `3`: chaser1
        - `4`: chaser2
    - (Adjust if needed, e.g., for both chasers in one cell.)

---

### **3.7 Legal Move Constraints**
- **Agents may move**: Up, Down, Left, Right, or Stay.
- Moves may **not** go outside the board boundaries or into an obstacle.
- **Stay** is only allowed when *all* four directional moves are blocked (edge or obstacle).

---

### **3.8 Chaser Overlap**
- Chaser1 and Chaser2 **may occupy the same cell**.
    - If both occupy a cell, show as `C` in the ASCII renderer.
- The Runner **may not** share a cell with either chaser except when being caught (game ends).

---

### **3.9 Win/Loss and Game End Conditions**
- **Chasers win**: If, after any agent’s move, the Runner shares a cell with either chaser, the game ends immediately.
- **Runner wins**:
    1. If the Runner is **completely surrounded** by obstacles/edges (cannot move and no chaser is present in that cell), the Runner wins immediately.
    2. If **300 turns** elapse without the Runner being caught, the Runner wins.

---

### **3.10 Initial Board Generation Algorithm**
**(For the coding agent or developer)**
1. Initialize a 15x15 empty board.
2. Randomly select valid agent positions, ensuring the Manhattan-distance constraint between the Runner and each Chaser is met.
3. For each agent, compute the forbidden zone: all cells within Chebyshev distance ≤ 1.
4. Compile a list of all legal, empty cells not in any forbidden zone.
5. Randomly select 5 unique cells from this list for obstacles.
6. Fill in the board with agents and obstacles according to above rules.

---

### **3.11 Reproducibility**
- All random choices (agent spawn, obstacle placement) **must** be reproducible given a fixed random seed.

---

### **3.12 ASCII Renderer Rules**
- Print the board after every move.
    - Label columns `a–o`, rows `1–15`.
    - Use:
        - `█` = obstacle
        - `R` = runner
        - `C1` = chaser1
        - `C2` = chaser2
        - `C` = both chasers overlap
        - `.` = empty cell

---

## 4 · Core Game Logic (board_state.py)

- Pure Python, no PettingZoo dependencies.
- Handles:
  - Board/grid state as a numpy array.
  - Agent spawning, move validation, obstacle placement.
  - Win/lose conditions, legal move generation.
- Expose:
  - Methods for resetting, stepping (moving agents), querying legal moves, and checking terminal state.
  - API to provide full observation (board + positions) for agents and RL.

---

## 5 · PettingZoo Environment (cgame_env.py)

- **Subclass `pettingzoo.AECEnv`.**
- Delegates all move, reset, and observation logic to `BoardState`.
- **Agents:**  
  - `["runner", "chaser1", "chaser2"]` (fixed order).
- **Action space:**  
  - Discrete(5): up, down, left, right, stay.
- **Observation space:**  
  - Full board matrix (numpy or dict encoding obstacles, agents).
- **Episode length:**  
  - Default 300 max turns (constant or CLI-override).
- Use `pettingzoo.test.api_test` in `test_env.py` to validate the environment.

---

## 6 · Agent Policies (agent_policies.py)

- Define a `BaseAgent` interface:
  ```python
  class BaseAgent:
      def decide_move(self, observation, legal_moves) -> int: ...
  ```
- Implement:
  - `RandomAgent`: uniformly random legal moves.
  - `ManualAgent`: takes user input via CLI.
  - (Optional) Hooks for learned agents (DQN, etc.).
- CLI/game can easily swap agent policies by injecting different objects.

---

## 7 · CLI & Board Rendering (cli_game.py, utils.py)

- **Initial Version:**  
  - The CLI game runner will allow for terminal play, with basic features such as:
    - Board visualization after each move (ASCII art).
    - Option for manual or random agent policies.
    - End-of-game result messages.

- **Future Development:**  
  - The CLI will be **expanded in a later phase** to include:
    - More detailed user prompts for custom board setup, agent configuration, and reproducibility (e.g., random seed entry).
    - Improved error handling and user feedback.
    - Enhanced visual feedback (optional: color, history tracking, statistics, etc.).
    - Modular user input code to allow easy addition of new agent types or rule modifications.

**Note:**  
*For this phase, the CLI (`cli_game.py`) will focus on providing basic playability and visualization.  
All advanced features, detailed user configuration, and interface improvements will be addressed in a subsequent development phase.*

---

## 8 · Factories & Utilities (utils.py)

- Provide helper functions to:
  - Spawn legal initial board/agent positions.
  - Generate legal moves for any agent.
  - Print and format board.
- All random elements (spawn, obstacles) to support a fixed seed for reproducibility.

---

## 9 · Testing & Debugging (test_env.py)

- Script to:
  - Run hundreds of random games, print average/median game length, win stats.
  - Run PettingZoo API compliance tests.

---

## 10 · Parameters & CLI Flags

- Board size (default 15×15; rarely changed).
- Max turns (default 300; can override).
- Random seed (optional).
- Agent policy selection (manual/random).
- (Optional) Debug flags (extra prints).

---

## 11 · Output

- After every move: updated ASCII board to terminal.
- End-of-game message: who won, at what turn, where.
- (Optional) Print summary stats in test runs.

---

## 12 · Environment & Dependencies

| Item           | Requirement |
|----------------|-------------|
| **Python**     | ≥ 3.10      |
| **PettingZoo** | Yes         |
| **Numpy**      | Yes         |
| **Rich**       | Optional (ASCII colors) |
| **typer/argparse** | CLI |
| **dataclasses, enum** | For clean code |
| **No persistence/networking in phase 1** | — |

---

## 13 · Deliverables

1. **`README.md`** with usage examples and agent swap instructions.
2. Docstrings for all classes and functions (PEP 257).
3. Clean, modular source code as above.

---

## 14 · Non-Goals

- No file/network I/O (beyond CLI) or GUI at this stage.
- No automated unit tests required for first version.
- **No advanced CLI features in the initial implementation; these will be specified and developed in detail in a later project phase.**

---

**End of PRD**  
*(This document establishes a clear modular structure, sets expectations for CLI development, and supports smooth expansion and RL integration.)*

---
