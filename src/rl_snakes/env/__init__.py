import gymnasium as gym

from rl_snakes.env.snake import SnakeEnv

gym.register(id="Snake-v0", entry_point="rl_snakes.env.snake:SnakeEnv")

__all__ = ["SnakeEnv"]
