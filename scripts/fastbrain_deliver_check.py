"""Check the one-pass delivery kernel against the event-buffer path, and time both.

Both brains start from the same random membrane state with no Poisson input, so
the dynamics are deterministic; after N steps the spike trains and voltages must
agree (up to floating-point summation order). Then the graph-replayed block is
timed for a realistic life batch.
Usage: python scripts/fastbrain_deliver_check.py [--batch 9] [--steps 400]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402

from flysim import connectome  # noqa: E402
from flysim.fastbrain import FastBrain  # noqa: E402
from flysim.physiology import DEFAULT, apply, neuron_meta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=9)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()
    con = connectome.load()
    model = apply(con, DEFAULT, neuron_meta(con))
    params = DEFAULT.lif()
    B = args.batch
    idx = torch.arange(10, device="cuda")
    brains = {name: FastBrain(model, B, params, idx, idx, idx[:0], steps=20, max_spikes=8192 * B,
                              max_events=2_000_000 * B, deliver=(name == "deliver")) for name in ("buffer", "deliver")}
    gen = torch.Generator(device="cuda").manual_seed(0)
    v0 = params.v_0 + torch.rand(con.n, B, device="cuda", generator=gen) * 7.2      # a few start above threshold
    import cupy
    cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream).use()
    spikes = {}
    for name, b in brains.items():
        b.reset_all()
        b.v.copy_(v0)
        total = torch.zeros(con.n, B, device="cuda")
        for s in range(args.steps):
            total += b._step(s % b.delay_steps).float()
        torch.cuda.synchronize()
        spikes[name] = total
        print(f"{name:8s} spikes {int(total.sum())}  overflow {float(b.overflow)}")
    d_sp = (spikes["buffer"] - spikes["deliver"]).abs().sum().item()
    d_v = (brains["buffer"].v - brains["deliver"].v).abs().max().item()
    print(f"spike count difference {d_sp:.0f}  max |dv| {d_v:.5f} mV")

    for name, b in brains.items():
        b.max_spikes, b.max_events = 256 * B, 30_000 * B
        b._alloc_events()
        b.capture()
        b.v.copy_(v0)
        b.rates.fill_(0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            b.run()
        torch.cuda.synchronize()
        print(f"{name:8s} block {(time.perf_counter() - t0) / 100 * 1000:6.2f} ms  (event buffer {b.max_events}, overflow {float(b.overflow)})")


if __name__ == "__main__":
    main()
