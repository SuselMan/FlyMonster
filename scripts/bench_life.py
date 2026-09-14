"""Speed of the life world only: N world steps with a frame every 5th, after a warm-up.
Usage: python scripts/bench_life.py [--flies 7] [--max-flies 9] [--steps 2000] [--no-deliver]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flysim.life.sim import Life, LifeConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--flies", type=int, default=7)
ap.add_argument("--max-flies", type=int, default=9)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--no-deliver", action="store_true")
args = ap.parse_args()
life = Life(LifeConfig(n_flies=args.flies, max_flies=args.max_flies, seed=3))
if args.no_deliver:
    life.brain.deliver = False
    life.brain.graph = None
for _ in range(500):
    life.step()
t0 = time.perf_counter()
for k in range(args.steps):
    life.step()
    if k % 5 == 0:
        life.frame()
wall = time.perf_counter() - t0
b = life.brain
print(f"flies {args.flies}/{args.max_flies} deliver={b.deliver}: speed {args.steps * 0.01 / wall:.2f}x  alive {len(life.ids)}  "
      f"spike buffer {b.max_spikes} event buffer {b.max_events}", flush=True)
