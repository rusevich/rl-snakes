import argparse

from rl_snakes.algos.vpg import VPG

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, nargs="+", default=[10], help="N for NxN, or W H")
    parser.add_argument("--total_episodes", type=int, default=-1, help="episodes to train for, -1 = forever")
    parser.add_argument("--render_frequency", type=int, default=1000, help="render in human mode every N episodes")
    parser.add_argument("--checkpoint_frequency", type=int, default=1000, help="save the policy every N episodes")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--ma_window", type=int, default=100, help="episodes in the moving average of returns")
    args = parser.parse_args()

    if len(args.size) > 2:
        parser.error("--size takes 1 or 2 values")
    if min(args.size) < 4:
        parser.error("--size: grid must be at least 4x4")
    args.size = args.size[0] if len(args.size) == 1 else tuple(args.size)
    if args.total_episodes < 1 and args.total_episodes != -1:
        parser.error("--total_episodes must be >= 1 or -1")
    if args.render_frequency < 1:
        parser.error("--render_frequency must be >= 1")
    if args.checkpoint_frequency < 1:
        parser.error("--checkpoint_frequency must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.ma_window < 1:
        parser.error("--ma_window must be >= 1")

    return args

def main():
    args = parse_args()
    algo = VPG(args)
    algo.train()


if __name__ == "__main__":
    main()