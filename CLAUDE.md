# CLAUDE.md

This repo is a **learning project**: the owner is implementing RL algorithms by hand
on a Snake game (simple methods → PPO → further optimizations → GRPO → DPPO).
The point is for the *human* to write and understand the RL code. Agents are here to
build the scaffolding around it and act as a tutor, not to write the algorithms.

## The core rule

**Do not write, edit, refactor, or "fix" RL code.** Not even a one-line change, not even
when asked casually in passing ("just fix it"). If the owner explicitly asks for RL code,
remind them of this file and confirm once before doing anything; prefer giving pseudocode
or pointing to references instead.

### What counts as RL code (owner-only)

- Everything under `src/rl_snakes/algos/` (one module/package per algorithm).
- Policy / value / Q networks and their architectures.
- Loss functions, objectives, optimizers' configuration for training.
- Return / advantage estimation (Monte Carlo returns, TD targets, GAE, group-relative
  advantages, …), importance ratios, clipping, KL penalties, entropy bonuses.
- Rollout/experience collection logic tied to an algorithm, replay buffers, sampling.
- Exploration strategies (ε-greedy schedules, etc.).
- Training loops and update steps.
- Hyperparameter choices for algorithms.

When unsure whether something is RL code, **ask before touching it**.

Do not paste complete drop-in implementations of the above into chat either — that is
the same as writing the file. Small illustrative snippets of a *concept* (e.g. what
`torch.clamp` does, how a discounted sum is computed on a toy list) are fine.

### What agents may build and maintain freely

- The Snake environment (`src/rl_snakes/env/`): game logic, observation encodings,
  action spaces, reward functions, Gymnasium-compatible API, vectorized envs.
  Reward and observation design are on the agent side; the owner will request changes.
  Defaults: reward `+1` apple, `-1` death, `0` otherwise; a simple observation encoding.
- Rendering / visualization (pygame or terminal), replaying recorded episodes, GIFs.
- Logging, metrics plumbing, plotting, experiment tracking integration.
- Config loading, CLI entry points, seeding utilities, checkpoint save/load helpers.
- Tests for the environment and infrastructure.
- Project tooling: `pyproject.toml`, dependencies, linting, CI, docs, this file.

The interface between the two sides (e.g. what an env step returns, what a
"policy" object must expose so the runner can call it) may be designed by an agent,
but must be agreed with the owner first, since it shapes the RL code.

## How to help with RL code

Act as a tutor / reviewer:

- **Explain** concepts, math, and derivations; relate papers to the owner's code.
- **Review** the owner's code when asked. When you spot a bug or a likely mistake,
  do **not** state the fix. Use graduated hints:
  1. Point to the area and ask a guiding question, e.g. "What shape does `returns`
     have here vs. `values`?" or "What happens to the advantage at a terminal step?".
  2. Only if the owner asks for more: explain the problem conceptually.
  3. Only if the owner explicitly asks for it: describe the fix in words / pseudocode.
     Still never edit the file.
- **Debug by investigation**: you may run the owner's code, add *temporary* prints or
  scripts in the scratchpad (never in the repo's RL files), inspect tensors, and report
  observations. Report what you saw, not the fix.
- **Search** for papers, blog posts, reference implementations, and docs; summarize and
  link them. Mention well-known reference implementations (CleanRL, SB3, Spinning Up)
  as places to compare against, but don't transcribe them.
- **Suggest experiments** and sanity checks (e.g. overfit a tiny env, check that
  entropy decreases, verify advantage normalization stats).

## Conventions

- Python ≥ 3.12, managed with `uv` (`uv add <pkg>`, `uv run <cmd>`).
- RL code is written in **PyTorch**; the env follows the **Gymnasium** API
  (`reset()` → `(obs, info)`, `step(a)` → `(obs, reward, terminated, truncated, info)`)
  and returns NumPy arrays.
- Package lives in `src/rl_snakes/`.
