"""Do taste and danger reach the mushroom-body dopamine neurons in this model?

Mushroom-body learning is three-factor: a Kenyon cell spike, a dopamine neuron (DAN) spike in the
same compartment, and the KC->MBON synapse between them weakens. Reward DANs (PAM cluster) are
driven by sugar in the fly, punishment DANs (PPL1 cluster) by bitter taste, shock, heat. Before
writing the rule we ask whether the connectome already does the first half: does sugar on the
labellum excite PAMs, does bitter excite PPL1s, and does an odor alone leave them quiet?

Stimuli (1 s each, MB physiology): nothing | sugar | bitter | water | fruit odor | fruit + sugar |
looming | JON-CE (dust). Reported: mean rate of PAM and PPL1 DANs, the DAN types most driven, and
MBON rates. Results -> results/dan_drive_scan.json.
Usage: python scripts/dan_drive_scan.py [--seconds 1.0] [--physiology mb|default]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from flysim import config, connectome, neurons  # noqa: E402
from flysim.brain import FlyBrain  # noqa: E402
from flysim.physiology import PRESETS, apply, neuron_meta  # noqa: E402
from flysim.senses import Olfaction  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--physiology", choices=list(PRESETS), default="mb")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    dev = torch.device(args.device)
    con = connectome.load()
    meta = neuron_meta(con)
    phys = PRESETS[args.physiology]
    model = apply(con, phys, meta)
    params = phys.lif()
    cls = meta.cell_class.fillna("").to_numpy()
    ct = meta.cell_type.fillna("").to_numpy()
    sub = meta.cell_sub_class.fillna("").to_numpy()
    dan = np.flatnonzero(cls == "DAN")
    pam = dan[np.char.startswith(ct[dan].astype(str), "PAM")]
    ppl1 = dan[np.char.startswith(ct[dan].astype(str), "PPL1")]
    mbon = np.flatnonzero(cls == "MBON")
    ids = lambda lst: np.array([con.index_of[i] for i in lst if i in con.index_of])
    groups = {"sugar": ids(neurons.SUGAR_GRN), "bitter": np.flatnonzero((cls == "gustatory") & (sub == "bitter")),
              "water": ids(neurons.WATER_GRN), "looming": np.flatnonzero(np.isin(ct, ["LC4", "LPLC2"])),
              "jon": ids(neurons.JON_CE)}
    conds = {"nothing": {}, "sugar": {"sugar": 150}, "bitter": {"bitter": 150}, "water": {"water": 200},
             "fruit": {"fruit": 0.6}, "fruit+sugar": {"fruit": 0.6, "sugar": 150}, "looming": {"looming": 150},
             "jon": {"jon": 150}}
    names = list(conds)
    B = len(names)
    olf = Olfaction(meta, B, dev)
    conc = torch.zeros(B, len(olf.names), 2, device=dev)
    order = list(groups)
    idx = np.concatenate([groups[g] for g in order])
    rates_g = torch.zeros(len(idx), B, device=dev)
    for j, name in enumerate(names):
        start = 0
        for g in order:
            n = len(groups[g])
            rates_g[start:start + n, j] = conds[name].get(g, 0)
            start += n
        if "fruit" in conds[name]:
            conc[j, olf.names.index("fruit")] = conds[name]["fruit"]
    input_idx = torch.cat([torch.tensor(idx, device=dev), olf.input_idx])
    brain = FlyBrain(model, batch=B, params=params, device=str(dev))
    counts = torch.zeros(con.n, B, device=dev)
    steps = int(args.seconds * 1000 / params.dt)
    r_olf = olf.rates(conc, params.dt * 20)
    for s in range(steps):
        if s % 20 == 0:
            r_olf = olf.rates(conc, params.dt * 20)
        counts += brain.step(input_idx, torch.cat([rates_g, r_olf]))
    hz = counts.cpu().numpy() / args.seconds
    out = {"args": vars(args), "conditions": {}}
    print(f"{'condition':12s} {'PAM Hz':>7s} {'PAM act':>7s} {'PPL1 Hz':>8s} {'PPL1 act':>8s} {'MBON Hz':>8s}  top DAN types")
    for j, name in enumerate(names):
        types = {}
        for t in sorted(set(ct[dan])):
            sel = dan[ct[dan] == t]
            types[t] = float(hz[sel, j].mean())
        top = sorted(types, key=lambda t: -types[t])[:4]
        rec = {"pam_hz": float(hz[pam, j].mean()), "pam_active": int((hz[pam, j] > 0).sum()),
               "ppl1_hz": float(hz[ppl1, j].mean()), "ppl1_active": int((hz[ppl1, j] > 0).sum()),
               "mbon_hz": float(hz[mbon, j].mean()), "dan_types_hz": types}
        out["conditions"][name] = rec
        print(f"{name:12s} {rec['pam_hz']:7.2f} {rec['pam_active']:4d}/{len(pam):<3d}{rec['ppl1_hz']:8.2f} {rec['ppl1_active']:4d}/{len(ppl1):<3d}"
              f" {rec['mbon_hz']:8.2f}  " + ", ".join(f"{t} {types[t]:.1f}" for t in top if types[t] > 0))
    path = config.RESULTS / "dan_drive_scan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
