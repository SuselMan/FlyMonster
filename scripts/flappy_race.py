"""Record the current best fly on long courses, for racing it in the browser.

Writes results/flappy_<run>/races/race_XX.json (all pipes + fly trace) and races/index.json.
Usage: python scripts/flappy_race.py --run first [--races 20] [--max-ticks 2400]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

from flysim import config, connectome  # noqa: E402
from flysim.flappy import FlappyConfig  # noqa: E402
from flysim.flappy_fly import FlappyFly, FlyParams  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="first")
    ap.add_argument("--races", type=int, default=20)
    ap.add_argument("--max-ticks", type=int, default=2400)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    out = config.RESULTS / f"flappy_{args.run}"
    state = json.loads((out / "state.json").read_text())
    params = FlyParams.from_vector(np.array(state["mean"]))
    cfg = FlappyConfig(max_ticks=args.max_ticks)
    n_pipes = int(cfg.max_ticks * cfg.speed / cfg.spacing) + 4
    seeds = [900000 + i for i in range(args.races)]

    t0 = time.perf_counter()
    fly = FlappyFly(connectome.load())
    games, traces = fly.play(seeds, [params] * len(seeds), cfg, record=True, n_pipes=n_pipes)
    (out / "races").mkdir(exist_ok=True)
    index = []
    for i, seed in enumerate(seeds):
        race = {"seed": seed, "gen": state["gen"] - 1, "cfg": cfg.__dict__,
                "params": {"gains": params.gains, "threshold": params.threshold},
                "pipes": games.pipes_json(i, n_pipes), "fly_score": int(games.score[i]),
                "fly_ticks": int(games.ticks[i]),
                "fly": [[s["y"], int(s["flap"])] for s in traces[i]]}
        name = f"race_{i:02d}.json"
        (out / "races" / name).write_text(json.dumps(race))
        index.append({"file": name, "fly_score": race["fly_score"], "fly_ticks": race["fly_ticks"]})
    (out / "races" / "index.json").write_text(json.dumps(index))
    print(f"{len(seeds)} races in {time.perf_counter() - t0:.0f} s; fly scores {sorted(games.score.tolist())}")


if __name__ == "__main__":
    main()
