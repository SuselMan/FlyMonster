"""Which descending neurons carry the side of an odor, and which respond to food/hunger?

Life flies steer only with DNa01/DNa02, which barely follow odor side. This scan
drives the same sensory channels the world uses (olfaction transducer, vision
salience, looming, sugar, touch, NPF hunger bias) with fixed stimuli and records
every descending neuron (super_class "descending", ~1300 cells).

For each condition, reps x brains get independent Poisson input; rates are
averaged over an early (0-0.5 s) and a late (0.5-2 s) window.
Results -> results/dn_scan.json (per neuron rates) and a printed summary:
  - odor side: neurons whose rate differs between odor-left and odor-right,
    grouped by cell type, with ipsi/contra consistency across both hemispheres;
  - food / hunger / takeoff candidates.
Usage: python scripts/dn_scan.py [--reps 6] [--seconds 2]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from flysim import config, connectome  # noqa: E402
from flysim.fastbrain import FastBrain  # noqa: E402
from flysim.life.sim import Life  # noqa: E402
from flysim.physiology import DEFAULT, apply, neuron_meta  # noqa: E402
from flysim.senses import Olfaction  # noqa: E402

HI, LO = 0.6, 0.15      # odor concentration at the near / far antenna (world plumes give similar ratios)
CONDITIONS = {
    "baseline": {},
    "fruit L": {"odor": ("fruit", HI, LO)}, "fruit R": {"odor": ("fruit", LO, HI)},
    "fruit both": {"odor": ("fruit", HI, HI)},
    "vinegar L": {"odor": ("vinegar", HI, LO)}, "vinegar R": {"odor": ("vinegar", LO, HI)},
    "fruit L hungry": {"odor": ("fruit", HI, LO), "hunger": 4.0}, "fruit R hungry": {"odor": ("fruit", LO, HI), "hunger": 4.0},
    "hungry": {"hunger": 4.0},
    "predator odor": {"odor": ("spider", HI, HI)}, "fly odor": {"odor": ("fly", HI, HI)},
    "sugar": {"sugar": 150.0}, "sugar hungry": {"sugar": 225.0, "hunger": 4.0},
    "loom L": {"loom_L": 180.0, "loom_R": 54.0}, "loom R": {"loom_L": 54.0, "loom_R": 180.0},
    "vis L": {"vis_L": 150.0}, "vis R": {"vis_R": 150.0},
    "touch L": {"touch_L": 100.0}, "touch R": {"touch_R": 100.0},
}
BLOCK_S = 0.010
EARLY_S = 0.5


class _Wiring:
    """Borrow Life's sensory wiring without building a world."""
    _wire = Life._wire

    def __init__(self, con, meta, dev):
        self.con, self.meta, self.dev = con, meta, dev
        self._wire()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=2.0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    dev = torch.device("cuda")
    con = connectome.load()
    meta = neuron_meta(con)
    model = apply(con, DEFAULT, meta)
    params = DEFAULT.lif()
    w = _Wiring(con, meta, dev)

    ct = meta.cell_type.fillna("").to_numpy()
    side = meta.side.fillna("").to_numpy()
    sup = meta.super_class.fillna("").to_numpy()
    dn = np.flatnonzero(sup == "descending")
    names = list(CONDITIONS)
    C, R = len(names), args.reps
    B = C * R
    olf = Olfaction(meta, B, dev)
    input_idx = torch.cat([w.sense_idx, olf.input_idx])
    brain = FastBrain(model, B, params, input_idx, torch.tensor(dn, device=dev), w.npf_idx,
                      steps=round(BLOCK_S * 1000 / params.dt), max_spikes=64 * B, max_events=11_000 * B)
    gi = {n: i for i, n in enumerate(w.group_names)}
    print(f"{len(dn)} descending neurons, {C} conditions x {R} reps = {B} brains", flush=True)

    # static sensory drive and odor concentrations per brain column
    drive = torch.zeros(len(w.group_names), B, device=dev)
    conc = torch.zeros(B, len(olf.names), 2, device=dev)
    bias = torch.zeros(B, device=dev)
    for c, name in enumerate(names):
        cols = slice(c * R, (c + 1) * R)
        for k, v in CONDITIONS[name].items():
            if k == "odor":
                o, left, right = v
                conc[cols, olf.names.index(o), 0] = left
                conc[cols, olf.names.index(o), 1] = right
            elif k == "hunger":
                bias[cols] = v
            else:
                drive[gi[k], cols] = v
    sense_rates = drive[w.sense_col]
    brain.bias.copy_(bias)

    blocks = round(args.seconds / BLOCK_S)
    early_blocks = round(EARLY_S / BLOCK_S)
    while True:
        brain.reset_all()
        olf.state.zero_()
        early = torch.zeros(len(dn), B, device=dev)
        late = torch.zeros(len(dn), B, device=dev)
        t0 = time.perf_counter()
        for k in range(blocks):
            brain.rates[:len(w.sense_idx)] = sense_rates
            brain.rates[len(w.sense_idx):] = olf.rates(conc, BLOCK_S * 1000)
            brain.run()
            (early if k < early_blocks else late).add_(brain.counts)
        torch.cuda.synchronize()
        if float(brain.overflow) == 0:
            break
        print("buffer overflow, doubling and repeating", flush=True)
        brain.max_spikes, brain.max_events = brain.max_spikes * 2, brain.max_events * 2
        brain._alloc_events()
        brain.overflow.zero_()
        brain.capture()
    print(f"simulated {args.seconds} s x {B} brains in {time.perf_counter() - t0:.1f} s", flush=True)

    early = (early / EARLY_S).cpu().numpy().reshape(len(dn), C, R)                        # Hz
    late = (late / (args.seconds - EARLY_S)).cpu().numpy().reshape(len(dn), C, R)
    out = {"conditions": names, "reps": R, "seconds": args.seconds, "early_s": EARLY_S,
           "neurons": [{"idx": int(i), "root_id": int(con.ids[i]) if hasattr(con, "ids") else None,
                        "type": ct[i], "side": side[i]} for i in dn],
           "early_hz": np.round(early.mean(2), 2).tolist(), "late_hz": np.round(late.mean(2), 2).tolist(),
           "early_sd": np.round(early.std(2), 2).tolist(), "late_sd": np.round(late.std(2), 2).tolist()}
    path = config.RESULTS / "dn_scan.json"
    path.write_text(json.dumps(out))
    print(f"saved {path}")
    summarize(names, dn, ct, side, early, late)


def _t(a, b):
    """Welch t statistic over reps (last axis)."""
    va, vb = a.var(-1, ddof=1), b.var(-1, ddof=1)
    return (a.mean(-1) - b.mean(-1)) / np.sqrt((va + vb) / a.shape[-1] + 1e-6)


def summarize(names, dn, ct, side, early, late):
    ix = {n: i for i, n in enumerate(names)}
    types, sides = ct[dn], side[dn]
    for win, X in (("early 0-0.5 s", early), ("late 0.5-2 s", late)):
        print(f"\n=== {win} ===")
        for pair in (("fruit L", "fruit R"), ("vinegar L", "vinegar R"), ("fruit L hungry", "fruit R hungry"),
                     ("loom L", "loom R"), ("vis L", "vis R"), ("touch L", "touch R")):
            a, b = X[:, ix[pair[0]]], X[:, ix[pair[1]]]
            t = _t(a, b)
            d = a.mean(-1) - b.mean(-1)
            # ipsi index: for a left neuron, stimulus-left minus stimulus-right; for a right neuron, the reverse
            ipsi = np.where(sides == "left", d, np.where(sides == "right", -d, np.nan))
            ipsi_t = np.where(sides == "left", t, np.where(sides == "right", -t, 0))
            strong = np.flatnonzero((np.abs(t) > 4) & (np.abs(d) > 3))
            print(f"\n{pair[0]} vs {pair[1]}: {len(strong)} neurons with |t|>4 and |diff|>3 Hz")
            by_type = {}
            for j in strong:
                by_type.setdefault(types[j] or "?", []).append(j)
            rows = []
            for ty, js in by_type.items():
                all_j = np.flatnonzero(types == ty)
                both = {sides[j] for j in js}
                rows.append((np.nanmean(np.abs(ipsi[js])), ty, len(js), len(all_j),
                             np.nanmean(ipsi_t[all_j]), "+".join(sorted(both)),
                             np.round(a.mean(-1)[js], 1).tolist(), np.round(b.mean(-1)[js], 1).tolist()))
            for m, ty, n, n_all, it, sd, ra, rb in sorted(rows, reverse=True)[:15]:
                print(f"  {ty:12s} {n}/{n_all} cells, sides {sd:10s} mean ipsi t {it:+5.1f}  "
                      f"{pair[0]} {ra}  {pair[1]} {rb}")
        base = X[:, ix["baseline"]]
        for cond in ("fruit both", "hungry", "fruit L hungry", "sugar", "sugar hungry", "fly odor", "predator odor"):
            a = X[:, ix[cond]]
            t = _t(a, base)
            d = a.mean(-1) - base.mean(-1)
            js = np.flatnonzero((t > 4) & (d > 5))
            js = js[np.argsort(-d[js])][:12]
            print(f"\n{cond} vs baseline, up: " + ", ".join(f"{types[j]}({sides[j][:1]}) {base.mean(-1)[j]:.0f}->{a.mean(-1)[j]:.0f}"
                                                         for j in js))


if __name__ == "__main__":
    main()
