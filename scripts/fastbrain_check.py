"""Compare FastBrain (CUDA graph) with FlyBrain: same responses, how much faster.

Drives sugar GRNs (-> MN9) and looming LC4/LPLC2 (-> giant fiber) in half of
the batch each, over 2 s, and reports readout rates and wall time per step.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from flysim import connectome, neurons  # noqa: E402
from flysim.brain import FlyBrain  # noqa: E402
from flysim.fastbrain import FastBrain  # noqa: E402
from flysim.physiology import DEFAULT, apply, neuron_meta  # noqa: E402

B, SECONDS = 32, 2.0


def main():
    con = connectome.load()
    meta = neuron_meta(con)
    model = apply(con, DEFAULT, meta)
    ct = meta.cell_type.fillna("").to_numpy()
    sugar = np.array(con.indices(i for i in neurons.SUGAR_GRN if i in con.index_of))
    loom = np.flatnonzero(np.isin(ct, ["LC4", "LPLC2"]))
    inp = torch.tensor(np.concatenate([sugar, loom]))
    read = torch.tensor([con.index_of[neurons.MN9], *np.flatnonzero(ct == "DNp01")])
    rates = torch.zeros(len(inp), B)
    rates[:len(sugar), : B // 2] = 150.0
    rates[len(sugar):, B // 2:] = 150.0
    p = DEFAULT.lif()
    blocks = int(SECONDS / 0.010)

    slow = FlyBrain(model, batch=B, params=p)
    counts = torch.zeros(len(read), B, device="cuda")
    ri, rr = inp.cuda(), rates.cuda()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(blocks * 20):
        counts += slow.step(ri, rr)[read.cuda()]
    torch.cuda.synchronize()
    slow_s = time.perf_counter() - t0
    slow_hz = counts.cpu().numpy() / SECONDS

    fast = FastBrain(model, B, p, inp, read, torch.tensor([], dtype=torch.long))
    fast.rates.copy_(rr)
    fast.capture()
    total = torch.zeros_like(fast.counts)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(blocks):
        fast.run()
        total += fast.counts
    torch.cuda.synchronize()
    fast_s = time.perf_counter() - t0
    fast_hz = total.cpu().numpy() / SECONDS

    for name, hz in (("FlyBrain", slow_hz), ("FastBrain", fast_hz)):
        print(f"{name:9}: MN9 sugar {hz[0, :B // 2].mean():6.1f} Hz, looming {hz[0, B // 2:].mean():5.1f} | "
              f"GF sugar {hz[1:, :B // 2].mean():5.1f}, looming {hz[1:, B // 2:].mean():6.1f} Hz")
    print(f"wall for {SECONDS} s of {B} brains: FlyBrain {slow_s:.2f} s, FastBrain {fast_s:.2f} s "
          f"(x{slow_s / fast_s:.1f}); overflow flag {float(fast.overflow)}")


if __name__ == "__main__":
    main()
