"""Do descending neurons steer by wind and head direction, and does odor gate it?

Real flies find odor sources mostly by turning upwind while they smell the odor
(odor-gated anemotaxis); steering towards a goal runs through the central complex
compass (E-PG -> ... -> PFL3 -> DNa02). This scan gives the brain the two inputs
our life flies lack: wind on the antennae (Johnston's organ, senses.Wind) and a
head-direction bump on the E-PG ring (senses.Compass), with and without fruit odor.

Grid: heading x egocentric wind direction x odor, every descending neuron recorded.
For each neuron the late-window rate is fit with a + b sin(rel) + c cos(rel) +
d sin(heading) + e cos(heading) separately with and without odor.
Upwind steering would show up as left-right DN differences (DNa01/DNa02 ...)
following sin(rel): wind from the left (rel = +90 deg) -> left turn.
Results -> results/wind_scan.json, summary printed.
Usage: python scripts/wind_scan.py [--angles 8] [--reps 2] [--seconds 1.5]
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
from flysim.physiology import DEFAULT, apply, neuron_meta  # noqa: E402
from flysim.senses import Compass, Olfaction, Wind  # noqa: E402

BLOCK_S = 0.010
EARLY_S = 0.5
ODOR = 0.6
HUNGER = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--wind", type=float, default=0.8)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    dev = torch.device("cuda")
    con = connectome.load()
    meta = neuron_meta(con)
    model = apply(con, DEFAULT, meta)
    params = DEFAULT.lif()
    ct = meta.cell_type.fillna("").to_numpy()
    side = meta.side.fillna("").to_numpy()
    dn = np.flatnonzero(meta.super_class.fillna("").to_numpy() == "descending")
    npf = np.flatnonzero(np.char.find(ct.astype(str), "NPF") >= 0)

    ang = np.arange(args.angles) * 2 * np.pi / args.angles
    grid = [(h, r, o) for o in (0, 1) for h in ang for r in ang]          # heading, wind-from (egocentric), odor
    C = len(grid)
    B = C
    olf = Olfaction(meta, B, dev)
    wind = Wind(meta, dev)
    compass = Compass(con, meta, dev)
    input_idx = torch.cat([olf.input_idx, wind.input_idx, compass.input_idx])
    brain = FastBrain(model, B, params, input_idx, torch.tensor(dn, device=dev), torch.tensor(npf, device=dev),
                      steps=round(BLOCK_S * 1000 / params.dt), max_spikes=64 * B, max_events=11_000 * B)
    heading = torch.tensor([g[0] for g in grid], dtype=torch.float32, device=dev)
    rel = torch.tensor([g[1] for g in grid], dtype=torch.float32, device=dev)
    odor = torch.tensor([g[2] for g in grid], dtype=torch.float32, device=dev)
    conc = torch.zeros(B, len(olf.names), 2, device=dev)
    conc[:, olf.names.index("fruit"), :] = (ODOR * odor)[:, None]
    n_olf, n_wind = len(olf.input_idx), len(wind.input_idx)
    brain.bias.fill_(HUNGER)
    static = torch.cat([wind.rates(torch.full((B,), args.wind, device=dev), rel), compass.rates(heading)])
    print(f"{len(dn)} DNs, {len(wind.input_idx)} JO wind neurons, {len(compass.input_idx)} E-PG; {C} conditions x {args.reps} reps")

    blocks, early_blocks = round(args.seconds / BLOCK_S), round(EARLY_S / BLOCK_S)
    late = torch.zeros(len(dn), B, args.reps, device=dev)
    t0 = time.perf_counter()
    rep = 0
    while rep < args.reps:
        brain.reset_all()
        olf.state.zero_()
        acc = torch.zeros(len(dn), B, device=dev)
        for k in range(blocks):
            brain.rates[:n_olf] = olf.rates(conc, BLOCK_S * 1000)
            brain.rates[n_olf:] = static
            brain.run()
            if k >= early_blocks:
                acc += brain.counts
        torch.cuda.synchronize()
        if float(brain.overflow) > 0:
            print("buffer overflow, doubling and repeating")
            brain.max_spikes, brain.max_events = brain.max_spikes * 2, brain.max_events * 2
            brain._alloc_events()
            brain.overflow.zero_()
            brain.capture()
            continue
        late[:, :, rep] = acc / (args.seconds - EARLY_S)
        rep += 1
    print(f"simulated in {time.perf_counter() - t0:.1f} s")
    late = late.cpu().numpy()
    out = {"grid": [[float(h), float(r), int(o)] for h, r, o in grid], "reps": args.reps, "wind": args.wind,
           "neurons": [{"idx": int(i), "type": ct[i], "side": side[i]} for i in dn],
           "late_hz": np.round(late, 1).tolist()}
    (config.RESULTS / "wind_scan.json").write_text(json.dumps(out))
    summarize(grid, dn, ct, side, late)


def fit(y, h, r):
    """Least squares y ~ a + b sin r + c cos r + d sin h + e cos h; returns coefs and R^2."""
    X = np.stack([np.ones_like(r), np.sin(r), np.cos(r), np.sin(h), np.cos(h)], 1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    r2 = 1 - res.var() / (y.var() + 1e-9)
    return coef, r2


def summarize(grid, dn, ct, side, late):
    g = np.array(grid)
    types, sides = ct[dn], side[dn]
    mean = late.mean(2)
    print("\n=== steering DNs: rate ~ a + b sin(wind rel) + c cos(wind rel) + d sin(heading) + e cos(heading) ===")
    for ty in ("DNa01", "DNa02", "DNb05", "DNa03", "DNa04", "DNa11", "DNg13", "DNp35", "DNp06", "MDN", "DNp01", "DNp02", "DNp04", "DNp11"):
        for j in np.flatnonzero(types == ty):
            for o in (0, 1):
                sel = g[:, 2] == o
                coef, r2 = fit(mean[j, sel], g[sel, 0], g[sel, 1])
                print(f"  {ty:6s} {sides[j][:1]} odor={o}: mean {coef[0]:6.1f}  wind sin {coef[1]:+6.1f} cos {coef[2]:+6.1f}"
                      f"  heading sin {coef[3]:+6.1f} cos {coef[4]:+6.1f}  R2 {r2:.2f}")
    print("\n=== strongest wind-direction (sin rel, left-right) tuning among all DNs, odor on vs off ===")
    rows = []
    for j in range(len(dn)):
        c0, _ = fit(mean[j, g[:, 2] == 0], g[g[:, 2] == 0, 0], g[g[:, 2] == 0, 1])
        c1, r1 = fit(mean[j, g[:, 2] == 1], g[g[:, 2] == 1, 0], g[g[:, 2] == 1, 1])
        rows.append((abs(c1[1]), j, c0, c1, r1))
    for _, j, c0, c1, r1 in sorted(rows, reverse=True)[:25]:
        print(f"  {types[j]:10s} {sides[j][:1]}  odor off: sin {c0[1]:+6.1f} cos {c0[2]:+6.1f} | odor on: sin {c1[1]:+6.1f}"
              f" cos {c1[2]:+6.1f} mean {c1[0]:6.1f} R2 {r1:.2f}")
    print("\n=== strongest head-direction tuning among DNs (odor on) ===")
    rows = []
    for j in range(len(dn)):
        c1, r1 = fit(mean[j, g[:, 2] == 1], g[g[:, 2] == 1, 0], g[g[:, 2] == 1, 1])
        rows.append((np.hypot(c1[3], c1[4]), j, c1, r1))
    for amp, j, c1, r1 in sorted(rows, reverse=True)[:12]:
        print(f"  {types[j]:10s} {sides[j][:1]}  heading amp {amp:5.1f} (sin {c1[3]:+.1f} cos {c1[4]:+.1f}) mean {c1[0]:.1f} R2 {r1:.2f}")


if __name__ == "__main__":
    main()
