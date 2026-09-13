"""Tune Flappy Fly input gains and decision threshold with the cross-entropy method.

Only 4 numbers evolve; the decision itself is the fly's DNb05 left/right.
Generation 0 is the untrained fly (all gains 1, threshold 0).

Outputs in results/flappy_<run>/: log.jsonl, replays/gen_XXXXX.json, status.json, state.json
Usage: python scripts/flappy_evolve.py --run first [--gens 60] [--resume]
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
from flysim.flappy_fly import SENSES, FlappyFly, FlyParams  # noqa: E402

SHOWCASE = [424242, 777777]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="first")
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--elite", type=int, default=3)
    ap.add_argument("--max-ticks", type=int, default=600)
    ap.add_argument("--mode", default="dnb05")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    out = config.RESULTS / f"flappy_{args.run}"
    (out / "replays").mkdir(parents=True, exist_ok=True)
    state_file = out / "state.json"
    if args.resume and state_file.exists():
        state = json.loads(state_file.read_text())
        print(f"resumed at generation {state['gen']}")
    else:
        state = {"gen": 0, "mean": FlyParams().to_vector().tolist(), "std": [0.5] * len(SENSES) + [2.0],
                 "best": None}
        (out / "log.jsonl").unlink(missing_ok=True)

    cfg = FlappyConfig(max_ticks=args.max_ticks)
    fly = FlappyFly(connectome.load(), mode=args.mode)
    rng = np.random.default_rng(state["gen"])

    while state["gen"] < args.gens:
        gen, t0 = state["gen"], time.perf_counter()
        mean, std = np.array(state["mean"]), np.array(state["std"])
        pop = mean + std * rng.normal(size=(args.pop, len(mean)))
        members = np.vstack([pop, mean[None]])   # last one: current mean
        seeds = [gen * 100 + j for j in range(args.games)]

        params = [FlyParams.from_vector(v) for v in members for _ in seeds] + \
                 [FlyParams.from_vector(mean)] * len(SHOWCASE)
        all_seeds = seeds * len(members) + SHOWCASE
        games, traces = fly.play(all_seeds, params, cfg, record=True)

        n = len(members) * len(seeds)
        fit = games.fitness()[:n].reshape(len(members), len(seeds)).mean(1)
        score = games.score[:n].reshape(len(members), len(seeds)).mean(1)
        elite = np.argsort(-fit[:args.pop])[:args.elite]
        state["mean"] = pop[elite].mean(0).tolist()
        state["std"] = np.maximum(pop[elite].std(0), [0.05] * len(SENSES) + [0.3]).tolist()

        center = FlyParams.from_vector(mean)
        rec = {
            "gen": gen, "seconds": round(time.perf_counter() - t0, 1),
            "center_score": float(score[-1]), "best_score": float(score[:args.pop].max()),
            "pop_score": float(score[:args.pop].mean()),
            "showcase_score": [int(s) for s in games.score[n:]],
            "gains": center.gains, "threshold": center.threshold, "max_ticks": cfg.max_ticks,
        }
        with open(out / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        replay = {"gen": gen, "cfg": cfg.__dict__, "params": {"gains": center.gains, "threshold": center.threshold},
                  "games": [{"seed": all_seeds[i], "score": int(games.score[i]), "ticks": int(games.ticks[i]),
                             "pipes": games.pipes_json(i), "trace": traces[i]}
                            for i in list(range(n))[-len(seeds):] + list(range(n, len(all_seeds)))]}
        (out / "replays" / f"gen_{gen:05d}.json").write_text(json.dumps(replay))
        if state["best"] is None or rec["center_score"] > state["best"]["score"]:
            state["best"] = {"gen": gen, "score": rec["center_score"]}
        print(f"gen {gen:3d} {rec['seconds']:5.0f}s  score: mean fly {rec['center_score']:5.2f}  "
              f"best member {rec['best_score']:5.2f}  population {rec['pop_score']:5.2f}  "
              f"showcase {rec['showcase_score']}  gains " +
              " ".join(f"{k}={v:.2f}" for k, v in center.gains.items()) + f" thr={center.threshold:.1f}")
        state["gen"] += 1
        state_file.write_text(json.dumps(state))
        (out / "status.json").write_text(json.dumps({"gen": gen, "gens": args.gens, "time": time.time(),
                                                     "seconds": rec["seconds"]}))


if __name__ == "__main__":
    main()
