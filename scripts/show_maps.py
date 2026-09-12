"""Print a few generated mazes with their stats.

Usage: python scripts/show_maps.py [seed ...] [--size 8] [--traps 4] [--loops 0.2]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flysim import world  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", nargs="*", type=int, default=[1, 42, 12345])
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--traps", type=int, default=4)
    ap.add_argument("--loops", type=float, default=0.2)
    args = ap.parse_args()

    cfg = world.MazeConfig(width=args.size, height=args.size, loops=args.loops, n_traps=args.traps)
    for seed in args.seeds:
        m = world.generate(seed, cfg)
        path = world.bfs(m.grid, m.start)[m.food]
        safe = world.bfs(m.grid, m.start, blocked=m.traps)[m.food]
        print(f"seed {seed}: path to food {path} tiles, safe path {safe} tiles, traps {len(m.traps)}")
        print(world.render(m))

        left, right = world.smell(m, m.start, m.heading)
        print(f"  at start: smell left {left:.3f} / right {right:.3f}")
        for name, ray in zip(("forward", "left", "right", "back"), world.vision(m, m.start, m.heading)):
            seen = [f"food@{ray.food_dist}"] if ray.food_dist else []
            seen += [f"trap@{ray.trap_dist}"] if ray.trap_dist else []
            wall = "clear" if ray.wall_dist == cfg.vision_range else f"wall in {ray.wall_dist}"
            print(f"  eye {name:7}: {wall} {' '.join(seen)}")
        print()

    # Same seed must give the same map.
    a, b = world.generate(seeds := args.seeds[0], cfg), world.generate(seeds, cfg)
    assert (a.grid == b.grid).all() and a.food == b.food and a.traps == b.traps
    print("determinism check: ok")


if __name__ == "__main__":
    main()
