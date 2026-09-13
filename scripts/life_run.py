"""Run the fly arena and record it for 1:1 playback in viewer/life.html.

Writes results/life_<run>/: meta.json, chunks/chunk_XXXXX.json (10 s each,
frames every 50 ms), events.jsonl, status.json.
Usage: python scripts/life_run.py --run first [--flies 24] [--minutes 60]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flysim import config  # noqa: E402
from flysim.life.sim import READOUT, WORLD_DT, Life, LifeConfig  # noqa: E402

FRAME_EVERY = 5      # world steps (50 ms)
CHUNK_S = 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="first")
    ap.add_argument("--flies", type=int, default=24)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    out = config.RESULTS / f"life_{args.run}"
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    for f in (out / "chunks").glob("*.json"):
        f.unlink()
    (out / "events.jsonl").unlink(missing_ok=True)

    cfg = LifeConfig(n_flies=args.flies, seed=args.seed)
    life = Life(cfg)
    a = cfg.arena
    (out / "meta.json").write_text(json.dumps({
        "arena": {"size": a.size, "spider": a.spider, "odor_sigma": a.odor_sigma, "day_length": a.day_length},
        "readout": list(READOUT), "frame_dt": FRAME_EVERY * WORLD_DT, "chunk_s": CHUNK_S,
        "physiology": cfg.physiology.__dict__, "flies": args.flies, "started": time.time(),
    }))

    frames, chunk, n_events, t0 = [], 0, 0, time.perf_counter()
    steps = int(args.minutes * 60 / WORLD_DT)
    for k in range(steps):
        life.step()
        if k % FRAME_EVERY == 0:
            frames.append(life.frame())
        if len(frames) * FRAME_EVERY * WORLD_DT >= CHUNK_S:
            (out / "chunks" / f"chunk_{chunk:05d}.json").write_text(json.dumps({"index": chunk, "frames": frames}))
            with open(out / "events.jsonl", "a", encoding="utf-8") as f:
                for e in life.events[n_events:]:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            n_events = len(life.events)
            wall = time.perf_counter() - t0
            (out / "status.json").write_text(json.dumps({
                "chunks": chunk + 1, "sim_s": round(life.t, 1), "wall_s": round(wall, 1),
                "speed": round(life.t / wall, 3), "alive": len(life.ids), "time": time.time()}))
            print(f"t={life.t:7.1f}s  wall {wall:7.1f}s  speed {life.t / wall:.2f}x  alive {len(life.ids)}  "
                  f"events {n_events}")
            frames, chunk = [], chunk + 1
        if not life.ids:
            print("all flies died")
            break


if __name__ == "__main__":
    main()
