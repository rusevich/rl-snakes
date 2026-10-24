"""Watch a trained policy play: `uv run snake-watch runs/<run>/policy.pt`.

The checkpoint must be a whole module saved with `torch.save(net, path)` that maps a
batch-free observation to action logits.
"""

import argparse
from pathlib import Path

import pygame
import torch

from rl_snakes.env import RecordGif, SnakeEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="policy saved with torch.save(net, path)")
    parser.add_argument("--size", type=int, nargs="+", default=[10], help="N for NxN, or W H")
    parser.add_argument("--obs_type", default="features", choices=["features", "grid"])
    parser.add_argument("--episodes", type=int, default=0, help="episodes to play, 0 = forever")
    parser.add_argument("--greedy", action="store_true", help="take the argmax action instead of sampling")
    parser.add_argument("--fps", type=int, default=SnakeEnv.metadata["render_fps"])
    parser.add_argument("--gif", action="store_true", help="also save each episode to <checkpoint dir>/gifs/")
    args = parser.parse_args()
    size = args.size[0] if len(args.size) == 1 else tuple(args.size[:2])

    net = torch.load(args.checkpoint, weights_only=False)
    net.eval()

    env = SnakeEnv(size=size, obs_type=args.obs_type, render_mode="human")
    env.metadata = {**env.metadata, "render_fps": args.fps}
    if args.gif:
        env = RecordGif(env, args.checkpoint.parent / "gifs" / f"{args.checkpoint.stem}_{{episode}}.gif")

    scores = []
    obs, _ = env.reset()
    while args.episodes == 0 or len(scores) < args.episodes:
        if any(event.type == pygame.QUIT for event in pygame.event.get()):
            break
        with torch.no_grad():
            logits = net(torch.from_numpy(obs))
        action = logits.argmax() if args.greedy else torch.distributions.Categorical(logits=logits).sample()
        obs, _, terminated, truncated, info = env.step(action.item())
        if terminated or truncated:
            scores.append(info["score"])
            print(f"episode {len(scores)}: score {info['score']} ({info['cause']}), mean {sum(scores) / len(scores):.2f}")
            obs, _ = env.reset()
    env.close()


if __name__ == "__main__":
    main()
