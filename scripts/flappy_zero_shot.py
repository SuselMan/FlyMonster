"""Does the fly brain play Flappy Fly with no learning at all?

Usage: python scripts/flappy_zero_shot.py [--games 16]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

from flysim import connectome  # noqa: E402
from flysim.flappy import Games  # noqa: E402
from flysim.flappy_fly import MODES, FlappyFly, FlyParams  # noqa: E402


def baseline(seeds, policy):
    g = Games(seeds)
    rng = np.random.default_rng(0)
    while g.alive.any():
        g.step(policy(g, rng))
    return g


def report(name, g, seconds=None):
    t = f"  ({seconds:.0f} s)" if seconds else ""
    print(f"{name:34} score mean {g.score.mean():5.2f} max {g.score.max():2d}  ticks mean {g.ticks.mean():5.0f}{t}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=16)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    seeds = list(range(args.games))

    report("never flap", baseline(seeds, lambda g, r: np.zeros(g.B, bool)))
    report("random 25%", baseline(seeds, lambda g, r: r.random(g.B) < 0.25))
    report("rule: flap if gap above", baseline(seeds, lambda g, r: g.y < g.next_pipe()[2] - 0.03))

    con = connectome.load()
    for mode in MODES:
        fly = FlappyFly(con, mode=mode)
        for thr in (0, 2, 5):
            t0 = time.perf_counter()
            g, traces = fly.play(seeds, [FlyParams(threshold=thr)] * len(seeds), record=True)
            report(f"fly brain, {mode}, threshold {thr}", g, time.perf_counter() - t0)
            flaps = np.mean([np.mean([s["flap"] for s in tr]) for tr in traces])
            above = [s for tr in traces for s in tr if s["gap_above"] > 0.2]
            below = [s for tr in traces for s in tr if s["gap_below"] > 0.2]
            fr = lambda xs: np.mean([s["flap"] for s in xs]) if xs else float("nan")
            print(f"{'':34} flaps {flaps:.0%} of ticks; when gap above: {fr(above):.0%}, when gap below: {fr(below):.0%}")


if __name__ == "__main__":
    main()
