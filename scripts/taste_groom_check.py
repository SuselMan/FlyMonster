"""Check the Shiu et al. 2024 feeding and grooming results under our physiology overrides.

Sugar -> MN9, bitter suppresses sugar -> MN9, water -> MN9; Johnston's organ
JON-F / JON-CE -> aBN1 (antennal grooming). Also: does our wind transducer
(JO-C/E wind neurons) drive aBN1, i.e. would wind make flies groom?
Usage: python scripts/taste_groom_check.py [--reps 4] [--seconds 1.5]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from flysim import connectome, neurons  # noqa: E402
from flysim.fastbrain import FastBrain  # noqa: E402
from flysim.physiology import DEFAULT, apply, neuron_meta  # noqa: E402
from flysim.senses import Wind  # noqa: E402

BLOCK_S = 0.010


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=1.5)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    dev = torch.device("cuda")
    con = connectome.load()
    meta = neuron_meta(con)
    model = apply(con, DEFAULT, meta)
    params = DEFAULT.lif()
    sub = meta.cell_sub_class.fillna("").to_numpy()
    side = meta.side.fillna("").to_numpy()
    ids = lambda lst: [con.index_of[i] for i in lst if i in con.index_of]
    groups = {"sugar": ids(neurons.SUGAR_GRN), "bitter": ids(neurons.BITTER_GRN), "water": ids(neurons.WATER_GRN),
              "JON_F": ids(neurons.JON_F), "JON_CE": ids(neurons.JON_CE),
              "bristle": np.flatnonzero((sub == "head bristle") & (side == "left")).tolist()}
    wind = Wind(meta, dev)
    conds = {
        "baseline": {}, "sugar 100": {"sugar": 100}, "sugar 100 + bitter 100": {"sugar": 100, "bitter": 100},
        "sugar 100 + bitter 200": {"sugar": 100, "bitter": 200}, "bitter 150": {"bitter": 150},
        "water 100": {"water": 100}, "water 200": {"water": 200}, "water 300": {"water": 300}, "water 400": {"water": 400}, "JON_F 100": {"JON_F": 100}, "JON_F 200": {"JON_F": 200},
        "JON_CE 50": {"JON_CE": 50}, "JON_CE 100": {"JON_CE": 100}, "JON_CE 150": {"JON_CE": 150}, "JON_CE 200": {"JON_CE": 200},
        "sugar 100 + JON_CE 150": {"sugar": 100, "JON_CE": 150}, "bristle L 100": {"bristle": 100}, "wind 0.8 front": {"wind": 0.8},
        "wind 0.8 left": {"wind": 0.8, "rel": np.pi / 2},
    }
    names = list(conds)
    R, C = args.reps, len(names)
    B = R * C
    order = list(groups)
    input_idx = torch.tensor(sum((groups[g] for g in order), []) + wind.input_idx.tolist(), device=dev)
    read = [neurons.MN9, neurons.ABN1]
    brain = FastBrain(model, B, params, input_idx, torch.tensor(ids(read), device=dev), torch.zeros(0, dtype=torch.long, device=dev),
                      steps=round(BLOCK_S * 1000 / params.dt), max_spikes=256 * B, max_events=40_000 * B)
    rates = torch.zeros(len(input_idx), B, device=dev)
    for c, name in enumerate(names):
        cols = slice(c * R, (c + 1) * R)
        start = 0
        for g in order:
            n = len(groups[g])
            rates[start:start + n, cols] = conds[name].get(g, 0)
            start += n
        if "wind" in conds[name]:
            wr = wind.rates(torch.full((R,), conds[name]["wind"], device=dev), torch.full((R,), conds[name].get("rel", 0.0), device=dev))
            rates[start:, cols] = wr
    brain.rates.copy_(rates)
    acc = torch.zeros(len(read), B, device=dev)
    blocks = round(args.seconds / BLOCK_S)
    for _ in range(blocks):
        brain.run()
        acc += brain.counts
    torch.cuda.synchronize()
    hz = (acc / args.seconds).cpu().numpy().reshape(len(read), C, R)
    print(f"overflow: {float(brain.overflow)}")
    print(f"{'condition':26s} {'MN9 Hz':>12s} {'aBN1 Hz':>12s}")
    for c, name in enumerate(names):
        print(f"{name:26s} {hz[0, c].mean():7.1f}±{hz[0, c].std():4.1f} {hz[1, c].mean():7.1f}±{hz[1, c].std():4.1f}")


if __name__ == "__main__":
    main()
