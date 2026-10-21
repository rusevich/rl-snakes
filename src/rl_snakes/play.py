"""Play Snake yourself with the arrow keys: `uv run snake-play --size 12`."""

import argparse

import pygame

from rl_snakes.env import SnakeEnv
from rl_snakes.env.snake import LEFT, RIGHT, STRAIGHT

KEYS = {pygame.K_UP: 0, pygame.K_RIGHT: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, nargs="+", default=[10], help="N for NxN, or W H")
    args = parser.parse_args()
    size = args.size[0] if len(args.size) == 1 else tuple(args.size[:2])

    env = SnakeEnv(size=size, render_mode="human")
    env.reset()
    while True:
        wanted = env.direction
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                env.close()
                return
            if event.type == pygame.KEYDOWN and event.key in KEYS:
                wanted = KEYS[event.key]

        # Convert the absolute key direction into a relative action; reversing is ignored.
        action = {0: STRAIGHT, 1: RIGHT, 3: LEFT}.get((wanted - env.direction) % 4, STRAIGHT)
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"score: {info['score']}")
            env.reset()


if __name__ == "__main__":
    main()
