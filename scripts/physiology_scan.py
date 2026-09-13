"""Scan physiology overrides against the regression assays.

Usage: python scripts/physiology_scan.py [--quick]
Writes results/physiology_scan.json
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flysim import assays, config, connectome  # noqa: E402
from flysim.physiology import Physiology, neuron_meta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    con = connectome.load()
    meta = neuron_meta(con)
    grid = [Physiology()] if args.quick else [
        Physiology(modulators_fast=m, al_exc_ln=k, std_u=u)
        for m, k, u in itertools.product((1.0, 0.0), (1.0, 0.5, 0.25, 0.1, 0.0), (0.0, 0.03))
    ]
    results = []
    keys = list(assays.THRESHOLDS)
    print(f"{'mod':>4} {'excLN':>5} {'STD':>4} | " + " ".join(f"{k[:11]:>11}" for k in keys) + " | pass")
    for phys in grid:
        t0 = time.perf_counter()
        m = assays.run(con, phys, meta)
        ok = assays.check(m)
        results.append({"physiology": phys.__dict__, "metrics": m, "pass": ok, "all_pass": all(ok.values())})
        cells = " ".join(f"{m[k]:>10.2f}{'' if ok[k] else '!'}" for k in keys)
        print(f"{phys.modulators_fast:>4} {phys.al_exc_ln:>5} {phys.std_u:>4} | {cells} | "
              f"{sum(ok.values())}/{len(ok)}  ({time.perf_counter() - t0:.0f} s)")
    (config.RESULTS / "physiology_scan.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
