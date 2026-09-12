"""Check that the whole-brain simulation runs and report its speed.

Stimulates sugar-sensing gustatory neurons and prints the most active neurons.
Usage: python scripts/smoke_test.py [--ms 100] [--batch 1] [--device cpu] [--dt 0.1]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402

from flysim import connectome, neurons  # noqa: E402
from flysim.brain import FlyBrain  # noqa: E402
from flysim.config import LIFParams  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=float, default=100.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dt", type=float, default=0.1, help="integration step, ms")
    args = ap.parse_args()

    t0 = time.perf_counter()
    con = connectome.load()
    print(f"connectome: {con.n} neurons, {con.weights._nnz()} connections "
          f"({time.perf_counter() - t0:.1f} s)")

    ann = connectome.annotations()
    ann = ann[~ann.index.duplicated()]
    stim = con.indices(i for i in neurons.SUGAR_GRN if i in con.index_of)
    print(f"stimulating {len(stim)} sugar GRNs at 150 Hz")

    brain = FlyBrain(con, batch=args.batch, device=args.device, params=LIFParams(dt=args.dt))
    idx = torch.tensor(stim, device=brain.device)
    rate = torch.full((len(stim), args.batch), 150.0, device=brain.device)

    steps = int(args.ms / brain.p.dt)
    counts = torch.zeros(con.n, args.batch, device=brain.device)
    if brain.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        counts += brain.step(idx, rate)
    if brain.device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    print(f"device {brain.device}, batch {args.batch}: {args.ms:.0f} ms simulated in {wall:.1f} s "
          f"({wall / (args.ms / 1000):.1f}x slower than real time per fly batch)")

    mean = counts.mean(1).cpu()
    rates = mean / (args.ms / 1000)
    print(f"neurons that spiked: {(mean > 0).sum().item()}")
    print(f"MN9 (proboscis motor neuron): {rates[con.index_of[neurons.MN9]]:.1f} Hz "
          f"(paper: activated by sugar)")
    top = torch.topk(rates, 15)
    print("most active (Hz, cell type):")
    for hz, idx in zip(top.values.tolist(), top.indices.tolist()):
        rid = int(con.ids[idx])
        ctype = ann["cell_type"].get(rid, "?") if rid in ann.index else "?"
        print(f"  {hz:7.1f}  {ctype}  {'(stimulated)' if idx in stim else ''}")


if __name__ == "__main__":
    main()
