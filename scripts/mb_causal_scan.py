"""Does a change of KC->MBON synapses change motor output in this model?

Gate for mushroom-body learning. For each MBON cell type, KC->MBON weights
are set to 0x (like aversive depression) and 2x, a fruit odor is presented,
and descending-neuron rates are compared to baseline across replicates
(Poisson noise). Positive control: direct MBON stimulation.

Writes results/mb_causal_scan.json (or --out). --kc-cholinergic / --pn-kc: the physiology of
scripts/mb_revive_scan.py (Kenyon cell outputs alive, projection neuron -> KC drive scaled).
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from flysim import config, connectome  # noqa: E402
from flysim.brain import FlyBrain  # noqa: E402
from flysim.physiology import DEFAULT, apply, neuron_meta  # noqa: E402
from flysim.senses import Olfaction  # noqa: E402

REPS = 8
MS = 1000.0
STEERING = ["DNa01", "DNa02", "DNa03", "DNb05", "MDN", "DNp09", "DNp01", "oviDNa_a", "oviDNb"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kc-cholinergic", action="store_true")
    ap.add_argument("--pn-kc", type=float, default=1.0)
    ap.add_argument("--out", default=str(config.RESULTS / "mb_causal_scan.json"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    con = connectome.load()
    meta = neuron_meta(con)
    phys = replace(DEFAULT, kc_cholinergic=True) if args.kc_cholinergic else DEFAULT
    base = apply(con, phys, meta)
    if args.pn_kc != 1.0:
        from mb_revive_scan import scaled
        base = scaled(base, meta, args.pn_kc, 1.0)
    print(f"physiology {phys}, PN->KC x{args.pn_kc}")
    cls = meta.cell_class.fillna("").to_numpy()
    ct = meta.cell_type.fillna("").to_numpy()
    side = meta.side.fillna("").to_numpy()
    dn = np.flatnonzero((meta.super_class.fillna("") == "descending").to_numpy())
    dn_name = [f"{ct[i]} {side[i][:1].upper()}" for i in dn]

    w = base.weights
    crow, col, val = w.crow_indices(), w.col_indices(), w.values()
    rows = torch.repeat_interleave(torch.arange(con.n), crow[1:] - crow[:-1])
    kc_pre = torch.from_numpy(cls == "Kenyon_Cell")[rows]

    def run(weights_val, stim="odor", stim_idx=None):
        model = connectome.Connectome(con.ids, torch.sparse_csr_tensor(crow, col, weights_val, w.shape), con.index_of)
        brain = FlyBrain(model, batch=REPS, params=phys.lif())
        dev = brain.device
        counts = torch.zeros(con.n, REPS, device=dev)
        if stim == "odor":
            olf = Olfaction(meta, REPS, dev)
            conc = torch.zeros(REPS, len(olf.names), 2, device=dev)
            conc[:, olf.names.index("fruit")] = 1.0
            for step in range(int(MS / brain.p.dt)):
                rates = olf.rates(conc, brain.p.dt) if step % 20 == 0 else rates
                counts += brain.step(olf.input_idx, rates)
        else:
            idx = torch.tensor(stim_idx, device=dev)
            rates = torch.full((len(stim_idx), REPS), 100.0, device=dev)
            for _ in range(int(MS / brain.p.dt)):
                counts += brain.step(idx, rates)
        return counts.cpu().numpy() / (MS / 1000)

    t0 = time.perf_counter()
    hz0 = run(val)
    kc = cls == "Kenyon_Cell"
    mbon_types = sorted({t for t in ct[cls == "MBON"] if t})
    print(f"baseline odor: KCs active {int((hz0[kc].mean(1) > 0).sum())}/{int(kc.sum())}, "
          f"MBONs active {int((hz0[cls == 'MBON'].mean(1) > 0).sum())}/{int((cls == 'MBON').sum())}, "
          f"DNs active {int((hz0[dn].mean(1) > 0).sum())} ({time.perf_counter() - t0:.0f} s)")

    def effect(hz):
        a, b = hz[dn], hz0[dn]
        diff = a.mean(1) - b.mean(1)
        sd = np.sqrt((a.var(1) + b.var(1)) / 2) + 1.0   # +1 Hz floor against zero-variance blowups
        return diff, diff / sd

    report = {"baseline": {"kc_active": int((hz0[kc].mean(1) > 0).sum())}, "plasticity": [], "stimulation": []}
    for t in mbon_types:
        target = torch.from_numpy(ct == t)[col] & kc_pre
        n_syn = int(target.sum())
        if n_syn == 0:
            continue
        for scale in (0.0, 2.0):
            v = val.clone()
            v[target] *= scale
            diff, d = effect(run(v))
            top = np.argsort(-np.abs(d))[:5]
            named = {dn_name[i]: round(float(diff[i]), 1) for i in range(len(dn)) if ct[dn[i]] in STEERING and abs(d[i]) > 1}
            rec = {"mbon": t, "scale": scale, "kc_mbon_synapses": n_syn, "max_abs_d": float(np.abs(d).max()),
                   "dn_changed_d>2": int((np.abs(d) > 2).sum()),
                   "top": [(dn_name[i], round(float(diff[i]), 1), round(float(d[i]), 1)) for i in top],
                   "steering_dn_changes": named}
            report["plasticity"].append(rec)
            print(f"{t:10} x{scale:<3} syn {n_syn:5}  DNs |d|>2: {rec['dn_changed_d>2']:3}  top: "
                  + ", ".join(f"{n} {dh:+.0f}Hz(d={dd:+.1f})" for n, dh, dd in rec["top"][:3])
                  + (f"  steering: {named}" if named else ""))

    # Positive control: drive each MBON type directly.
    for t in mbon_types:
        idx = np.flatnonzero(ct == t).tolist()
        diff, d = effect(run(val, stim="mbon", stim_idx=idx))
        top = np.argsort(-np.abs(d))[:3]
        report["stimulation"].append({"mbon": t, "dn_changed_d>2": int((np.abs(d) > 2).sum()),
                                      "top": [(dn_name[i], round(float(diff[i]), 1)) for i in top]})
    print("direct MBON stimulation, DNs changed (|d|>2):",
          {r["mbon"]: r["dn_changed_d>2"] for r in report["stimulation"]})
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
