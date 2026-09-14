"""Where does the time of a life world step go, and which neurons make the brain expensive?

1. CPU phases of Life.step (sensory rates, olfaction, wind/compass, brain wait, body,
   arena, frame) with the GPU synchronised inside the brain phase only.
2. The brain block alone for the current event buffer size and for smaller ones
   (the fill kernel and index_add_ run over the whole fixed buffer every step).
3. Actual spikes and synaptic events per step (eager replay of the same state),
   and the neurons that generate most events (spikes x out-degree).
Usage: python scripts/profile_life.py [--flies 6] [--max-flies 8] [--steps 1500]
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from flysim.fastbrain import FastBrain  # noqa: E402
from flysim.life.sim import Life, LifeConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flies", type=int, default=6)
    ap.add_argument("--max-flies", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    life = Life(LifeConfig(n_flies=args.flies, max_flies=args.max_flies, seed=3))
    b = life.brain
    assert isinstance(b, FastBrain)

    # --- 1. phases ------------------------------------------------------------
    acc = defaultdict(float)

    def timed(obj, name, key, sync=False):
        f = getattr(obj, name)

        def wrap(*a, **k):
            if sync:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = f(*a, **k)
            if sync:
                torch.cuda.synchronize()
            acc[key] += time.perf_counter() - t0
            return r
        setattr(obj, name, wrap)

    for _ in range(300):                       # warm up, let buffers settle
        life.step()
    print(f"warm: buffers max_spikes {b.max_spikes}, max_events {b.max_events}, alive {len(life.ids)}")
    for name in ("_sensory_rates", "_olfaction", "_wind_and_compass", "_body", "_predators", "_hatch", "_check_overflow"):
        timed(life, name, name)
    timed(life, "_brain_hz", "_brain_hz (incl. GPU wait)")
    timed(life.arena, "update", "arena.update")
    t0 = time.perf_counter()
    frames = 0
    for k in range(args.steps):
        life.step()
        if k % 5 == 0:
            tf = time.perf_counter()
            life.frame()
            acc["frame (every 5th step)"] += time.perf_counter() - tf
            frames += 1
    total = time.perf_counter() - t0
    print(f"\n=== {args.steps} world steps ({args.steps * 0.01:.0f} s simulated) in {total:.1f} s -> speed {args.steps * 0.01 / total:.2f}x, alive {len(life.ids)} ===")
    for key, v in sorted(acc.items(), key=lambda kv: -kv[1]):
        print(f"  {key:30s} {v / args.steps * 1000:7.2f} ms/step  {v / total * 100:5.1f}%")
    other = total - sum(acc.values())
    print(f"  {'(rest of step)':30s} {other / args.steps * 1000:7.2f} ms/step  {other / total * 100:5.1f}%   budget for 1x: 10 ms/step")

    # --- 2. brain block alone vs event buffer size --------------------------
    print("\n=== brain block alone (graph replay), by event buffer size ===")
    state = [t.clone() for t in (b.v, b.g, b.refrac, b.spike_buf)]
    E0, S0 = b.max_events, b.max_spikes
    for frac in (1.0, 0.5, 0.25, 0.125):
        b.max_events, b.max_spikes = max(1024, int(E0 * frac)), max(64, int(S0 * frac))
        b._alloc_events()
        b.capture()
        for dst, src in zip((b.v, b.g, b.refrac, b.spike_buf), state):
            dst.copy_(src)
        b.overflow.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 200
        for _ in range(n):
            b.run()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n
        print(f"  events {b.max_events:>9d} spikes {b.max_spikes:>6d}: {dt * 1000:6.2f} ms/block  overflow={float(b.overflow) > 0}")
    b.max_events, b.max_spikes = E0, S0
    b._alloc_events()
    b.capture()
    for dst, src in zip((b.v, b.g, b.refrac, b.spike_buf), state):
        dst.copy_(src)

    # --- 3. actual activity (eager, same state and inputs) ------------------
    print("\n=== actual activity per 0.5 ms step (eager replay of 100 blocks) ===")
    n, B = b.n, b.batch
    deg = (b.crow[1:n + 1] - b.crow[:n]).float()
    spikes_per_neuron = torch.zeros(n, device=b.dev)
    if b.use_kernel:
        import cupy
        cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream).use()
    per_step_spikes, per_step_events = [], []
    for _ in range(100):
        for s in range(b.steps):
            arriving = b.spike_buf[s % b.delay_steps]
            cnt_sp = arriving.sum()
            per_step_spikes.append(cnt_sp)
            per_step_events.append((arriving.float().sum(1) * deg).sum())
            spikes_per_neuron += arriving.float().sum(1)
            b._step(s % b.delay_steps)
    sp = torch.stack(per_step_spikes).float().cpu().numpy()
    ev = torch.stack(per_step_events).cpu().numpy()
    print(f"  spikes/step: mean {sp.mean():.0f}  p99 {np.percentile(sp, 99):.0f}  max {sp.max():.0f}   (buffer {b.max_spikes})")
    print(f"  events/step: mean {ev.mean():.0f}  p99 {np.percentile(ev, 99):.0f}  max {ev.max():.0f}   (buffer {b.max_events})")
    secs = 100 * b.steps * b.p.dt * 1e-3 * B
    rate = (spikes_per_neuron / secs).cpu().numpy()
    share = (spikes_per_neuron * deg).cpu().numpy()
    share = share / share.sum()
    meta = life.meta
    ct = meta.cell_type.fillna("").to_numpy()
    cls = meta.super_class.fillna("").to_numpy()
    print(f"  neurons that fired at all: {(rate > 0).sum()}  >50 Hz: {(rate > 50).sum()}  >150 Hz: {(rate > 150).sum()}")
    order = np.argsort(-share)
    print(f"  top 1% of neurons make {share[order[:n // 100]].sum() * 100:.0f}% of events; top 100 make {share[order[:100]].sum() * 100:.0f}%")
    print("  top event makers (rate Hz per fly, out-degree, share of all events):")
    for i in order[:25]:
        print(f"    {ct[i] or '?':18s} {cls[i]:22s} {rate[i]:6.1f} Hz  deg {int(deg[i]):5d}  {share[i] * 100:5.2f}%")
    by_type = defaultdict(float)
    for i in order[:3000]:
        by_type[(ct[i] or "?", cls[i])] += share[i]
    print("  by cell type:")
    for (t, c), v in sorted(by_type.items(), key=lambda kv: -kv[1])[:15]:
        print(f"    {t:18s} {c:22s} {v * 100:5.1f}%")


if __name__ == "__main__":
    main()
