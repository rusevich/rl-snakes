from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Clockwise order, so turning right is +1 and turning left is -1.
DIRECTIONS = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)])  # up, right, down, left
STRAIGHT, RIGHT, LEFT = 0, 1, 2
TURNS = {STRAIGHT: 0, RIGHT: 1, LEFT: -1}


class SnakeEnv(gym.Env):
    """Snake on a `width x height` grid.

    Actions (relative to the current heading): 0 = straight, 1 = turn right, 2 = turn left.
    Rewards: +1 for eating an apple, -1 for dying, 0 otherwise.

    Observations (`obs_type`):
      - "features": 11 binary values —
          danger [straight, right, left],
          heading one-hot [up, right, down, left],
          apple is [up, right, down, left] of the head.
      - "grid": (3, height, width) channels — body, head, apple.

    The episode is truncated if the snake goes `max_idle_steps` steps without eating
    (default: width * height), so a looping policy can't run forever.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        size: int | tuple[int, int] = 10,
        obs_type: str = "features",
        max_idle_steps: int | None = None,
        render_mode: str | None = None,
    ):
        self.width, self.height = (size, size) if isinstance(size, int) else size
        assert self.width >= 4 and self.height >= 4, "grid must be at least 4x4"
        assert obs_type in ("features", "grid"), f"unknown obs_type {obs_type!r}"
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.obs_type = obs_type
        self.max_idle_steps = max_idle_steps or self.width * self.height
        self.render_mode = render_mode
        self._renderer = None

        self.action_space = spaces.Discrete(3)
        if obs_type == "features":
            self.observation_space = spaces.Box(0, 1, shape=(11,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(0, 1, shape=(3, self.height, self.width), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cx, cy = self.width // 2, self.height // 2
        self.direction = 1  # facing right
        self.snake = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])  # head first
        self.score = 0
        self.idle_steps = 0
        self._place_apple()
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info()

    def step(self, action):
        self.direction = (self.direction + TURNS[int(action)]) % 4
        head = self._next_cell(self.direction)
        eats = head == self.apple

        if not eats:
            tail = self.snake.pop()  # the tail moves away, so the head may enter its cell
        if self._collides(head):
            if not eats:
                self.snake.append(tail)  # keep the final frame intact for rendering
            reward, terminated = -1.0, True
        else:
            self.snake.appendleft(head)
            reward, terminated = 0.0, False
            self.idle_steps += 1
            if eats:
                reward = 1.0
                self.score += 1
                self.idle_steps = 0
                terminated = not self._place_apple()  # board is full: the snake won

        truncated = not terminated and self.idle_steps >= self.max_idle_steps
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, truncated, self._info()

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from rl_snakes.env.render import SnakeRenderer

            self._renderer = SnakeRenderer(self.width, self.height, self.render_mode, self.metadata["render_fps"])
        return self._renderer.draw(self.snake, self.apple, self.direction, self.score)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # --- helpers -------------------------------------------------------------

    def _next_cell(self, direction: int) -> tuple[int, int]:
        dx, dy = DIRECTIONS[direction]
        x, y = self.snake[0]
        return x + int(dx), y + int(dy)

    def _collides(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return not (0 <= x < self.width and 0 <= y < self.height) or cell in self.snake

    def _place_apple(self) -> bool:
        """Put the apple on a random free cell. Returns False if there is no free cell."""
        occupied = np.zeros((self.height, self.width), dtype=bool)
        for x, y in self.snake:
            occupied[y, x] = True
        free = np.argwhere(~occupied)
        if len(free) == 0:
            self.apple = None
            return False
        y, x = free[self.np_random.integers(len(free))]
        self.apple = (int(x), int(y))
        return True

    def _obs(self) -> np.ndarray:
        if self.obs_type == "grid":
            return self._grid_obs()
        return self._feature_obs()

    def _feature_obs(self) -> np.ndarray:
        # Danger is checked as if the tail stayed put, which is a slight overestimate.
        danger = [self._collides(self._next_cell((self.direction + TURNS[a]) % 4)) for a in (STRAIGHT, RIGHT, LEFT)]
        heading = np.eye(4)[self.direction]
        hx, hy = self.snake[0]
        ax, ay = self.apple if self.apple is not None else (hx, hy)
        apple = [ay < hy, ax > hx, ay > hy, ax < hx]
        return np.concatenate([danger, heading, apple]).astype(np.float32)

    def _grid_obs(self) -> np.ndarray:
        grid = np.zeros((3, self.height, self.width), dtype=np.float32)
        for x, y in self.snake:
            grid[0, y, x] = 1
        hx, hy = self.snake[0]
        grid[1, hy, hx] = 1
        if self.apple is not None:
            grid[2, self.apple[1], self.apple[0]] = 1
        return grid

    def _info(self) -> dict:
        return {"score": self.score, "length": len(self.snake)}
