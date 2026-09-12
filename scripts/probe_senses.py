"""Does each sensory group reach the descending neurons, and how fast?

Drives one feature group at a time for 200 ms and reports readout activity.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402

from flysim import body, connectome  # noqa: E402
from flysim.brain import FlyBrain  # noqa: E402
from flysim.config import LIFParams  # noqa: E402


def main():
    con = connectome.load()
    wiring = body.build(con)
    features = [("nose", f) for f in body.NOSE_FEATURES] + [("eye", f) for f in body.EYE_FEATURES
                                                            if not f.startswith("hear")]
    features += [("nose", "ALL"), ("eye", "ALL")]  # every feature at once
    brain = FlyBrain(con, batch=len(features), params=LIFParams(dt=0.5))
    io = body.IO(wiring, brain.device)

    rates = torch.zeros(len(io.input_idx), len(features), device=brain.device)
    for b, (role, name) in enumerate(features):
        drive = torch.zeros(1, len(io.maps[role].T), device=brain.device)
        if name == "ALL":
            drive[:] = 1.0
        else:
            drive[0, (body.NOSE_FEATURES if role == "nose" else body.EYE_FEATURES).index(name)] = 1.0
        rates[:, b] = io.rates(role, drive)[:, 0]

    dt = brain.p.dt
    first_dn = torch.full((len(features),), float("nan"))
    counts = torch.zeros(len(io.readout_idx), len(features), device=brain.device)
    t0 = time.perf_counter()
    for step in range(int(200 / dt)):
        spikes = brain.step(io.input_idx, rates)
        dn = spikes[io.readout_idx].float()
        counts += dn
        fired = (dn.sum(0) > 0).cpu() & first_dn.isnan()
        first_dn[fired] = step * dt
    print(f"simulated 200 ms x {len(features)} in {time.perf_counter() - t0:.1f} s")
    # Input off: activity should fade. If it stays high, the input triggers runaway.
    after = torch.zeros(len(features), device=brain.device)
    for _ in range(int(100 / dt)):
        after += brain.step().sum(0)
    after = after.cpu() / int(100 / dt)

    pooled = io.pooled(counts).cpu()
    active_dn = (counts > 0).sum(0).cpu()
    print(f"{'feature':22} {'first DN spike':>14} {'active DNs':>10} {'pool activity':>13} {'spikes/step after off':>22}")
    for b, (role, name) in enumerate(features):
        t = "never" if first_dn[b].isnan() else f"{first_dn[b]:.1f} ms"
        flag = "  RUNAWAY" if after[b] > 20 else ""
        print(f"{role + '.' + name:22} {t:>14} {int(active_dn[b]):>10} {pooled[b].sum():13.2f} {after[b]:22.1f}{flag}")


if __name__ == "__main__":
    main()
