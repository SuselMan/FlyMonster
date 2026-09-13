"""Run the fly world and record it for 1:1 playback in viewer/life.html.

Writes results/life_<run>/: meta.json, chunks/chunk_XXXXX.json (10 s each,
frames every 50 ms), events.jsonl, status.json. Reads commands.jsonl
(appended by the viewer, e.g. dropping fruit) once per simulated second.
status.json carries world counters (flights, long flights, water crossings,
droppings, webs built, ant trips, ...).
Usage: python scripts/life_run.py --run first [--flies 24] [--minutes 60]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flysim import config  # noqa: E402
from flysim.life.sim import GENES, READOUT, SENSES, WORLD_DT, Life, LifeConfig  # noqa: E402

FRAME_EVERY = 5      # world steps (50 ms)
CHUNK_S = 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="first")
    ap.add_argument("--flies", type=int, default=6)
    ap.add_argument("--max-flies", type=int, default=None, help="brain slots (population cap), default flies + 2")
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--as-fast-as-possible", action="store_true", help="do not hold the simulation to real time")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    out = config.RESULTS / f"life_{args.run}"
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    for f in (out / "chunks").glob("*.json"):
        f.unlink()
    for name in ("events.jsonl", "commands.jsonl"):
        (out / name).unlink(missing_ok=True)

    cfg = LifeConfig(n_flies=args.flies, max_flies=args.max_flies or args.flies + 2, seed=args.seed)
    life = Life(cfg)
    (out / "meta.json").write_text(json.dumps({
        "arena": life.arena.static(), "readout": list(READOUT), "genes": list(GENES), "senses": list(SENSES),
        "frame_dt": FRAME_EVERY * WORLD_DT, "chunk_s": CHUNK_S, "hatch_time": cfg.hatch_time,
        "lifespan": cfg.lifespan, "physiology": cfg.physiology.__dict__, "flies": args.flies, "max_flies": cfg.max_flies, "started": time.time(),
    }))

    frames, chunk, n_events, commands_done, t0 = [], 0, 0, 0, time.perf_counter()
    steps = int(args.minutes * 60 / WORLD_DT)
    per_second = int(round(1.0 / WORLD_DT))
    for k in range(steps):
        if k % per_second == 0:
            commands_done = life.arena.apply_commands(out / "commands.jsonl", life.t, commands_done)
            if not args.as_fast_as_possible:
                ahead = life.t - (time.perf_counter() - t0)    # never run ahead of the wall clock
                if ahead > 0.05:
                    time.sleep(ahead)
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
                "speed": round(life.t / wall, 3), "alive": len(life.ids), "eggs": len(life.eggs),
                "births": life.births, "counters": life.frame()["counters"], "time": time.time()}))
            print(f"t={life.t:7.1f}s  wall {wall:7.1f}s  speed {life.t / wall:.2f}x  alive {len(life.ids)}  "
                  f"events {n_events}  {life.frame()['counters']}")
            frames, chunk = [], chunk + 1
        if not life.ids and k % (per_second * 60) == 0:
            print("no flies alive (eggs: %d); the world keeps running for viewer-released flies" % len(life.eggs))


if __name__ == "__main__":
    main()
