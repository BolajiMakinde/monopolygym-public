# Monopoly Gym Environment

This repository provides a Gym-compatible Monopoly environment for running simulations and tests.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running Tests](#running-tests)
- [Launching the Environment](#launching-the-environment)

---

## Overview

The **Monopoly Gym Environment** (`monopoly_gym`) implements:

- A Gym-compatible `MonopolyEnvironment` class under `monopoly_gym/env.py`.
- A complete board representation (`monopoly_gym/gym/board.py`).
- State and action logic (`monopoly_gym/gym/state.py`, `monopoly_gym/gym/action.py`, etc.).
- Built-in players (e.g., `MaskedRandomPlayer`, `HumanPlayer`, etc.) and a simple CLI renderer (`monopoly_gym/renderer.py`).

You can run automated tests against this environment (via `pytest`) or launch a live game from the command line.

---

## Prerequisites

1. **Python 3.8+** (ensure `python --version` is 3.8 or higher).
2. **pipenv** installed globally:

   ```bash
   pip install pipenv
   ```

---

## Installation

1. **Clone the repository** (if you haven’t already):

   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```

2. **Create a new Pipenv environment** and install dependencies:

   ```bash
   pipenv install --dev
   ```

   - This reads `Pipfile` (and `Pipfile.lock`) to install required packages (e.g., `gym`, `pygame`, `pytest`, etc.).
   - The `--dev` flag ensures testing dependencies (`pytest`, etc.) are included.

3. **Activate the Pipenv shell**:

   ```bash
   pipenv shell
   ```

   Once inside the shell, your prompt will change to indicate you’re using the virtual environment.

---

## Running Tests

All tests are located under `tests/` (or sometimes alongside modules) and are written for `pytest`. To run the full test suite:

```bash
pytest
```

Or, equivalently:

```bash
python -m pytest
```

This command will:

- Discover any files matching `test_*.py` or `*_test.py`.
- Execute all assertions.
- Report pass/fail status.

---

## Launching the Environment

You can start a simple game from the command line. By default, the script will:

1. Instantiate a `MonopolyEnvironment`.
2. Add three `MaskedRandomPlayer` agents (AI players).
3. Begin a GUI game loop (using `pygame` for rendering).

### Command

```bash
python -m monopoly_gym.env
```

- **Requirements**:

  - Within your Pipenv shell, run the above command.
  - A Pygame window will open showing the Monopoly board.
  - Three AI players (named `AI 1`, `AI 2`, `AI 3`) will take turns automatically.
  - Logs print to stdout and (optionally) a file named `monopoly_game_main.log`.

### Customizing Players

If you want to use your own players (e.g., human CLI or a different bot), edit the `__main__` block in `monopoly_gym/env.py`:

```python
if __name__ == "__main__":
    max_turns = 300
    game = MonopolyEnvironment(
        max_turns=max_turns,
        use_render=True,
        enable_general_log=True,
        general_log_file="monopoly_game_main.log",
        enable_timestamped_log=True,
        timestamped_log_dir="game_logs"
    )

    # Example: Replace MaskedRandomPlayer with HumanPlayer or another agent
    players = [
        HumanPlayer(name="You", mgn_code="H1"),
        MaskedRandomPlayer(name="AI 1", mgn_code="A1", action_space_type=ActionSpaceType.HIERARCHICAL),
        MaskedRandomPlayer(name="AI 2", mgn_code="A2", action_space_type=ActionSpaceType.HIERARCHICAL),
    ]
    for player in players:
        game.add_player(player=player)

    game.play()
    game.close()
```

- **`use_render=True`** → opens the Pygame window.
- **`use_render=False`** → skips rendering (headless mode).

---

## Notes

- **Logging**:

  - By default, general logs go to `monopoly_game_main.log`.
  - Enabling `timestamped_log_dir` creates a new file per run under `game_logs/`.

- **Exiting**:

  - Close the Pygame window or press `ESC` to quit.
  - The script will clean up handlers and exit gracefully.

---

With these steps, you should be able to:

1. Install all dependencies with `pipenv`.
2. Run the automated tests via `pytest`.
3. Launch a live Monopoly Gym game using `python -m monopoly_gym.env`.

Enjoy experimenting with the Monopoly Gym environment!
