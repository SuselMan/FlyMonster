"""Does the brain steer by humidity the way it steers by odor?

Life flies find apples by odor: the left-right difference of fruit ORN input ends up as a
left-right difference of the evolved steering DN pair (DNa02, DNg99; scripts/evolve_steer.py).
To find water they need the same for humidity. FlyWire has the sacculus hygrosensory neurons
(cell_class hygrosensory: moist cells HRN_VP5, dry cells HRN_VP4, humid TRN_VP1m, ...), so the question is
whether a left-right difference of moist-cell input reaches the same DNs with the same sign.

Conditions (steady input, late-window rates): none | fruit L>R | fruit R>L | humid L>R |
humid R>L | humid both | dry L>R. For every descending-neuron type the L-R rate difference is
reported, the steering turn implied by the evolved weights, and the whole-brain rate (runaway check).
Results -> results/hygro_scan.json.
Usage: python scripts/hygro_scan.py [--reps 4] [--seconds 1.5] [--hi 0.6] [--lo 0.4]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from flysim import config, connectome  # noqa: E402
from flysim.fastbrain import FastBrain  # noqa: E402
from flysim.life.sim import STEER_EVOLVED, STEER_TYPES  # noqa: E402
from flysim.physiology import DEFAULT, apply, neuron_meta  # noqa: E402
from flysim.senses import HUMIDITY, Olfaction  # noqa: E402

BLOCK_S = 0.010
EARLY_S = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--hi", type=float, default=0.6)
    ap.add_argument("--lo", type=float, default=0.4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    dev = torch.device(args.device)
    cuda = dev.type == "cuda"
    con = connectome.load()
    meta = neuron_meta(con)
    model = apply(con, DEFAULT, meta)
    params = DEFAULT.lif()
    ct = meta.cell_type.fillna("").to_numpy()
    side = meta.side.fillna("").to_numpy()
    dn = np.flatnonzero(meta.super_class.fillna("").to_numpy() == "descending")

    hi, lo = args.hi, args.lo
    conds = {
        "none": {},
        "fruit L>R": {"fruit": (hi, lo)}, "fruit R>L": {"fruit": (lo, hi)},
        "humid L>R": {"humid": (hi, lo)}, "humid R>L": {"humid": (lo, hi)}, "humid both": {"humid": (hi, hi)},
        "dry L>R": {"dry": (hi, lo)},
    }
    names = list(conds)
    R, C = args.reps, len(names)
    B = R * C
    cpu = torch.device("cpu")
    olf = Olfaction(meta, B, cpu)
    hyg = Olfaction(meta, B, cpu, odorants=HUMIDITY, tau_adapt_ms=6000.0, gamma=0.5)
    conc_o = torch.zeros(B, len(olf.names), 2)
    conc_h = torch.zeros(B, len(hyg.names), 2)
    for c, name in enumerate(names):
        cols = slice(c * R, (c + 1) * R)
        for odor, (l, r) in conds[name].items():
            tgt, nm = (conc_o, olf.names) if odor in olf.names else (conc_h, hyg.names)
            tgt[cols, nm.index(odor), 0] = l
            tgt[cols, nm.index(odor), 1] = r
    input_idx = torch.cat([olf.input_idx, hyg.input_idx]).to(dev)
    read_idx = torch.tensor(dn, device=dev)
    brain = FastBrain(model, B, params, input_idx, read_idx, torch.zeros(0, dtype=torch.long, device=dev),
                      steps=round(BLOCK_S * 1000 / params.dt), max_spikes=512 * B, max_events=60_000 * B,
                      device=str(dev), use_kernel=cuda)
    run = brain.run if cuda else brain.run_eager
    blocks = round(args.seconds / BLOCK_S)
    early = round(EARLY_S / BLOCK_S)
    late = torch.zeros(len(dn), B, device=dev)
    for b in range(blocks):
        rates = torch.cat([olf.rates(conc_o, BLOCK_S * 1000), hyg.rates(conc_h, BLOCK_S * 1000)])
        brain.rates.copy_(rates.to(dev))
        run()
        if b >= early:
            late += brain.counts
    if cuda:
        torch.cuda.synchronize()
    hz = (late / (args.seconds - EARLY_S)).cpu().numpy()                     # (dn, B)
    print(f"overflow: {brain.overflow_kind.cpu().numpy().tolist()}")
    types = sorted({ct[i] for i in dn if ct[i]})
    out = {"args": vars(args), "conditions": {}}
    hdr = f"{'condition':12s} " + " ".join(f"{t:>9s}" for t in STEER_TYPES[:6]) + f" {'turn':>7s} {'DN Hz':>7s}"
    print(hdr)
    for c, name in enumerate(names):
        cols = slice(c * R, (c + 1) * R)
        rec = {}
        for t in types:
            l = hz[np.isin(dn, np.flatnonzero((ct == t) & (side == "left")))][:, cols]
            r = hz[np.isin(dn, np.flatnonzero((ct == t) & (side == "right")))][:, cols]
            if len(l) and len(r):
                rec[t] = {"L": float(l.mean()), "R": float(r.mean()), "diff": float(l.mean() - r.mean()),
                          "diff_sd": float((l.mean(0) - r.mean(0)).std())}
        turn = sum(w * rec[t]["diff"] for t, w in STEER_EVOLVED.items() if t in rec) / 50.0
        mean_dn = float(hz[:, cols].mean())
        out["conditions"][name] = {"types": rec, "turn": turn, "dn_hz": mean_dn}
        print(f"{name:12s} " + " ".join(f"{rec[t]['diff']:+9.2f}" if t in rec else f"{'-':>9s}" for t in STEER_TYPES[:6])
              + f" {turn:+7.3f} {mean_dn:7.2f}")
    # which DN types carry humidity side most strongly
    hl = out["conditions"]["humid L>R"]["types"]
    hr = out["conditions"]["humid R>L"]["types"]
    fl = out["conditions"]["fruit L>R"]["types"]
    side_code = {t: (hl[t]["diff"] - hr[t]["diff"]) / 2 for t in hl if t in hr}
    top = sorted(side_code, key=lambda t: -abs(side_code[t]))[:12]
    print("\nhumidity side (mean of L>R and mirrored R>L, Hz), with the fruit side for comparison:")
    for t in top:
        print(f"  {t:10s} humid {side_code[t]:+7.2f}   fruit {fl[t]['diff']:+7.2f}   sd {hl[t]['diff_sd']:.2f}")
    out["humidity_side"] = side_code
    path = config.RESULTS / "hygro_scan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
