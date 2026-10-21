# rl-snakes

Learning RL algorithms by implementing them by hand on Snake: vanilla policy gradient →
PPO → further optimizations → GRPO → DPPO.

## Setup

```bash
uv sync
uv run snake-play --size 12   # play it yourself: arrow keys, Esc to quit
```

## The environment

`SnakeEnv` follows the [Gymnasium](https://gymnasium.farama.org/) API.

```python
from rl_snakes.env import SnakeEnv

env = SnakeEnv(size=10)                  # 10x10 grid
env = SnakeEnv(size=(16, 10))            # width x height
env = SnakeEnv(size=10, obs_type="grid") # grid observation instead of features
```

Or through `gym.make`:

```python
import gymnasium as gym
import rl_snakes.env  # registers "Snake-v0"

env = gym.make("Snake-v0", size=12)
```

### Arguments

| Argument         | Default      | Meaning                                                    |
| ---------------- | ------------ | ---------------------------------------------------------- |
| `size`           | `10`         | `N` for an `N x N` grid, or `(width, height)`. Minimum 4.  |
| `obs_type`       | `"features"` | `"features"` or `"grid"` (see below).                      |
| `max_idle_steps` | `None`       | Truncate after this many steps without eating. `None` = `width * height`. |
| `render_mode`    | `None`       | `None`, `"human"` (window) or `"rgb_array"` (frames).      |

### Basic loop

```python
env = SnakeEnv(size=10)
obs, info = env.reset(seed=0)

done = False
while not done:
    action = env.action_space.sample()  # random agent
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(info)  # {'score': 2, 'length': 5}
env.close()
```

### Actions

`Discrete(3)`, relative to where the snake is heading:

| Action | Meaning    |
| ------ | ---------- |
| `0`    | straight   |
| `1`    | turn right |
| `2`    | turn left  |

The snake can never reverse into itself.

### Observations

**`"features"`**: `float32` vector of shape `(11,)`, every value 0 or 1:

| Index | Meaning                                               |
| ----- | ----------------------------------------------------- |
| 0–2   | danger (wall or body) straight / right / left         |
| 3–6   | heading is up / right / down / left (one-hot)         |
| 7–10  | apple is up / right / down / left of the head         |

**`"grid"`**: `float32` array of shape `(3, height, width)`. The channels are
body (including the head), head and apple.

### Rewards and episode end

| Event                               | Reward | Result                                 |
| ----------------------------------- | ------ | -------------------------------------- |
| eats an apple                       | `+1`   | snake grows                            |
| hits a wall or itself               | `-1`   | `terminated = True`                    |
| fills the whole board               | `+1`   | `terminated = True` (it won)           |
| `max_idle_steps` without eating     | `0`    | `truncated = True` (cut off, not dead) |
| anything else                       | `0`    |                                        |

`info` always contains `score` (apples eaten) and `length` (snake length).

### Rendering

Watch in a window. Each `step` and `reset` draws a frame at 10 FPS:

```python
env = SnakeEnv(size=10, render_mode="human")
obs, info = env.reset()
for _ in range(200):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

Or grab frames as arrays, e.g. to make a GIF:

```python
env = SnakeEnv(size=10, render_mode="rgb_array")
env.reset(seed=0)
frame = env.render()  # np.ndarray, shape (H, W, 3), dtype uint8
```

### Vectorized envs

Run several envs at once, with observations stacked along the first axis:

```python
import gymnasium as gym

envs = gym.vector.SyncVectorEnv([lambda: SnakeEnv(size=10) for _ in range(8)])
obs, info = envs.reset(seed=0)  # obs.shape == (8, 11)
obs, rewards, terminated, truncated, info = envs.step(envs.action_space.sample())
```

Finished envs reset automatically on the next `step` (Gymnasium's default
`NEXT_STEP` autoreset). That step returns the first observation of the new episode, with
`reward = 0` and `terminated = truncated = False`, and the action passed for that env is
ignored. Keep this in mind when collecting rollouts.

## Layout

```
src/rl_snakes/
  env/     Snake environment and renderer
  algos/   RL algorithms, written by hand
  play.py  play the game with the keyboard
```
