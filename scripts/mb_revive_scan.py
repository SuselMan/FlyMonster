"""Can the mushroom body be brought to life without waking the antennal-lobe avalanche?

scripts/mb_causal_scan.py showed that KC->MBON plasticity changes nothing in this model because
almost nothing reaches the MBONs: under our physiology (weakened antennal-lobe local neurons) an
odor activates ~500 of 5177 Kenyon cells only weakly and the MBONs are nearly silent. Learning in
the fly works by depressing MBON output, so a silent MBON cannot learn.

Grid: projection neuron -> Kenyon cell drive x {1, 2, 3, 4, 6} and APL -> Kenyon cell inhibition
x {1, 0.5, 0}. For each condition two odors (fruit, vinegar; life transducer) run 1 s. Reported:
KCs active (target: 5-10 % per odor, sparse), KC/MBON pattern correlation between the two odors
(low = separable), MBON rates (must be non-zero), DN rates, and late whole-brain activity after
the odor (runaway). CPU is enough (small batch). Results -> results/mb_revive_scan.json.
Usage: python scripts/mb_revive_scan.py [--seconds 1.0] [--conc 0.6] [--kc-cholinergic] [--pn-kc 1,2] [--apl-kc 1]
"""
import argparse
from dataclasses import replace
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from flysim import config, connectome  # noqa: E402
from flysim.brain import FlyBrain  # noqa: E402
from flysim.physiology import DEFAULT, apply, neuron_meta  # noqa: E402
from flysim.senses import Olfaction  # noqa: E402

PN_KC = (1.0, 2.0, 3.0, 4.0, 6.0)
APL_KC = (1.0, 0.5, 0.0)


def scaled(base: connectome.Connectome, meta, pn_kc: float, apl_kc: float) -> connectome.Connectome:
    w = base.weights
    crow, col, val = w.crow_indices(), w.col_indices(), w.values().clone()
    pre = torch.repeat_interleave(torch.arange(base.n), crow[1:] - crow[:-1])
    cls = meta.cell_class.fillna("").to_numpy()
    ct = meta.cell_type.fillna("").to_numpy()
    kc_post = torch.from_numpy(cls == "Kenyon_Cell")[col]
    pn_pre = torch.from_numpy(cls == "ALPN")[pre]
    apl_pre = torch.from_numpy(ct == "APL")[pre]
    val[pn_pre & kc_post & (val > 0)] *= pn_kc
    val[apl_pre & kc_post] *= apl_kc
    return connectome.Connectome(base.ids, torch.sparse_csr_tensor(crow, col, val, w.shape), base.index_of)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--conc", type=float, default=0.6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--kc-cholinergic", action="store_true", help="Physiology(kc_cholinergic=True): KC outputs alive")
    ap.add_argument("--pn-kc", default=",".join(str(v) for v in PN_KC))
    ap.add_argument("--apl-kc", default=",".join(str(v) for v in APL_KC))
    ap.add_argument("--out", default=str(config.RESULTS / "mb_revive_scan.json"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    dev = torch.device(args.device)
    con = connectome.load()
    meta = neuron_meta(con)
    phys = replace(DEFAULT, kc_cholinergic=True) if args.kc_cholinergic else DEFAULT
    base = apply(con, phys, meta)
    params = phys.lif()
    cls = meta.cell_class.fillna("").to_numpy()
    ct = meta.cell_type.fillna("").to_numpy()
    kc = np.flatnonzero(cls == "Kenyon_Cell")
    mbon = np.flatnonzero(cls == "MBON")
    dan = np.flatnonzero(cls == "DAN")
    dn = np.flatnonzero(meta.super_class.fillna("").to_numpy() == "descending")
    olf = Olfaction(meta, 2, dev)
    conc = torch.zeros(2, len(olf.names), 2, device=dev)
    conc[0, olf.names.index("fruit")] = args.conc
    conc[1, olf.names.index("vinegar")] = args.conc
    steps = int(args.seconds * 1000 / params.dt)
    after = int(300 / params.dt)
    out = {"args": vars(args), "kc": int(len(kc)), "mbon": int(len(mbon)), "conditions": []}
    print(f"{'PN->KC':>6s} {'APL':>4s} | {'KC act %':>8s} {'KC Hz':>6s} {'KC corr':>7s} | {'MBON act':>8s} {'MBON Hz':>7s} "
          f"{'MBON corr':>9s} | {'DAN Hz':>6s} {'DN Hz':>6s} {'late/step':>9s} {'s':>4s}")
    for pn_kc in [float(v) for v in args.pn_kc.split(",")]:
        for apl_kc in [float(v) for v in args.apl_kc.split(",")]:
            t0 = time.time()
            brain = FlyBrain(scaled(base, meta, pn_kc, apl_kc), batch=2, params=params, device=str(dev))
            counts = torch.zeros(con.n, 2, device=dev)
            rates = None
            for s in range(steps):
                if s % 20 == 0:
                    rates = olf.rates(conc, params.dt * 20)
                counts += brain.step(olf.input_idx, rates)
            late = torch.zeros(2, device=dev)
            for _ in range(after):
                late += brain.step().sum(0)
            hz = counts.cpu().numpy() / args.seconds
            late = (late / after).cpu().numpy()

            def corr(idx):
                a, b = hz[idx, 0], hz[idx, 1]
                if a.std() == 0 or b.std() == 0:
                    return float("nan")
                return float(np.corrcoef(a, b)[0, 1])

            rec = {"pn_kc": pn_kc, "apl_kc": apl_kc,
                   "kc_active_frac": [float((hz[kc, j] > 0).mean()) for j in range(2)],
                   "kc_hz": float(hz[kc].mean()), "kc_corr": corr(kc),
                   "mbon_active": [int((hz[mbon, j] > 0).sum()) for j in range(2)],
                   "mbon_hz": float(hz[mbon].mean()), "mbon_corr": corr(mbon),
                   "mbon_types_hz": {t: float(hz[mbon][ct[mbon] == t].mean()) for t in sorted(set(ct[mbon])) if t},
                   "dan_hz": float(hz[dan].mean()), "dn_hz": float(hz[dn].mean()),
                   "late_spikes_per_step": float(late.max()), "seconds": round(time.time() - t0, 1)}
            out["conditions"].append(rec)
            print(f"{pn_kc:6.1f} {apl_kc:4.1f} | {100 * np.mean(rec['kc_active_frac']):8.1f} {rec['kc_hz']:6.2f} "
                  f"{rec['kc_corr']:7.2f} | {np.mean(rec['mbon_active']):8.1f} {rec['mbon_hz']:7.2f} {rec['mbon_corr']:9.2f} | "
                  f"{rec['dan_hz']:6.2f} {rec['dn_hz']:6.2f} {rec['late_spikes_per_step']:9.2f} {rec['seconds']:4.0f}")
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
