"""Classical conditioning in the whole-brain model: does the mushroom body learn?

Protocol (Tully & Quinn style, one brain per row of the batch):
  pre-test   odor A 1 s, pause, odor B 1 s                 -> MBON and DN responses before
  training   n x [odor A 1 s, dopamine drive in its 2nd half]   PAM (reward), PPL1 (punishment) or none
  post-test  odor A, odor B again                          -> responses after
Groups: reward (PAM DANs driven), punishment (PPL1 driven), control (odor only), each `--reps` times.
Reported per group: the plastic multipliers per MBON type (memory), MBON response to A and B before
and after (a learned change must be specific to A), and the steering / escape DN rates to A.

Runs the MB physiology (living mushroom body) on FastBrain with per-brain plasticity
(flysim/plasticity.py). Results -> results/mb_learning_check.json.
Usage: python scripts/mb_learning_check.py [--reps 4] [--trainings 3] [--eta 0.02] [--dan-hz 100]
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
from flysim.physiology import PRESETS, apply, neuron_meta  # noqa: E402
from flysim.plasticity import MushroomBody  # noqa: E402
from flysim.senses import Olfaction  # noqa: E402

BLOCK_S = 0.010
DN_TYPES = ("DNa02", "DNg99", "DNb05", "MDN", "DNp01", "DNp09", "DNa01")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--trainings", type=int, default=3)
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--dan-hz", type=float, default=100.0)
    ap.add_argument("--conc", type=float, default=0.6)
    ap.add_argument("--physiology", choices=list(PRESETS), default="mb")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    dev = torch.device(args.device)
    cuda = dev.type == "cuda"
    con = connectome.load()
    meta = neuron_meta(con)
    phys = PRESETS[args.physiology]
    model = apply(con, phys, meta)
    params = phys.lif()
    ct = meta.cell_type.fillna("").to_numpy()
    side = meta.side.fillna("").to_numpy()
    mb = MushroomBody(model, meta, dev, raw=con, eta=args.eta)
    pam = mb.dan_idx[np.char.startswith(mb.dan_type.astype(str), "PAM")]
    ppl1 = mb.dan_idx[np.char.startswith(mb.dan_type.astype(str), "PPL1")]
    print(f"plastic synapses {len(mb.syn)}, MBONs {len(mb.mbon_idx)}, DANs {len(mb.dan_idx)} (PAM {len(pam)}, PPL1 {len(ppl1)}), "
          f"MBONs with a compartment DAN set: {int((mb.n_dan_of_mbon > 0).sum())}")

    groups = ["reward", "punishment", "control"]
    R = args.reps
    B = R * len(groups)
    group_of = np.repeat(np.arange(len(groups)), R)
    cpu = torch.device("cpu")
    olf = Olfaction(meta, B, cpu)
    A, Bo = olf.names.index("fruit"), olf.names.index("vinegar")
    dan_in = torch.tensor(np.concatenate([pam, ppl1]), device=dev)
    input_idx = torch.cat([olf.input_idx.to(dev), dan_in])
    n_o = len(olf.input_idx)
    read_idx = torch.tensor(np.concatenate([mb.kc_idx, mb.dan_idx, mb.mbon_idx,
                                            np.flatnonzero(np.isin(ct, DN_TYPES))]), device=dev)
    k_kc, k_dan, k_mbon = len(mb.kc_idx), len(mb.dan_idx), len(mb.mbon_idx)
    dn_local = np.flatnonzero(np.isin(ct, DN_TYPES))
    brain = FastBrain(model, B, params, input_idx, read_idx, torch.zeros(0, dtype=torch.long, device=dev),
                      steps=round(BLOCK_S * 1000 / params.dt), max_spikes=512 * B, max_events=60_000 * B,
                      device=str(dev), use_kernel=cuda)
    mb.attach(brain)
    run = brain.run if cuda else brain.run_eager

    rows_r = torch.tensor(group_of == groups.index("reward"), device=dev)
    rows_p = torch.tensor(group_of == groups.index("punishment"), device=dev)

    def play(seconds: float, odor: int | None, dan: bool = False, record: bool = False):
        """Run `seconds` with an odor at both antennae; with `dan`, the reward rows get PAM drive and the
        punishment rows PPL1 drive. Returns summed readout counts when `record`."""
        acc = torch.zeros(len(read_idx), B, device=dev) if record else None
        conc = torch.zeros(B, len(olf.names), 2)
        if odor is not None:
            conc[:, odor, :] = args.conc
        rates = torch.zeros(len(input_idx), B, device=dev)
        if dan:
            rates[n_o:n_o + len(pam)][:, rows_r] = args.dan_hz
            rates[n_o + len(pam):][:, rows_p] = args.dan_hz
        for _ in range(round(seconds / BLOCK_S)):
            rates[:n_o] = olf.rates(conc, BLOCK_S * 1000).to(dev)
            brain.rates.copy_(rates)
            run()
            mb.step(brain.counts[:k_kc], brain.counts[k_kc:k_kc + k_dan], BLOCK_S)
            if record:
                acc += brain.counts
        return acc

    def test(label):
        out = {}
        for name, odor in (("A", A), ("B", Bo)):
            olf.state.zero_()                   # no carry-over of receptor adaptation into the test
            play(2.0, None)
            c = play(1.0, odor, record=True)
            play(2.0, None)
            hz = c.cpu().numpy()
            rec = {}
            for g, gname in enumerate(groups):
                cols = group_of == g
                mbon_hz = hz[k_kc + k_dan:k_kc + k_dan + k_mbon][:, cols].mean(1)
                dn = hz[k_kc + k_dan + k_mbon:][:, cols].mean(1)
                dns = {}
                for t in DN_TYPES:
                    for s in ("left", "right"):
                        sel = (ct[dn_local] == t) & (side[dn_local] == s)
                        if sel.any():
                            dns[f"{t} {s[0].upper()}"] = float(dn[sel].mean())
                rec[gname] = {"mbon_mean_hz": float(mbon_hz.mean()),
                              "mbon_by_type": {t: float(mbon_hz[mb.mbon_type == t].mean()) for t in sorted(set(mb.mbon_type)) if t},
                              "kc_active": float((hz[:k_kc][:, cols] > 0).mean()), "dn": dns}
            out[name] = rec
        print(f"{label}: MBON mean Hz to A / B: " + "  ".join(
            f"{g} {out['A'][g]['mbon_mean_hz']:.2f}/{out['B'][g]['mbon_mean_hz']:.2f}" for g in groups))
        return out

    t0 = time.time()
    torch.manual_seed(0)
    play(1.0, None)                             # settle
    pre = test("pre ")
    # which KCs each odor recruits (control rows, pre-test): for the specificity of the memory
    kc_a = torch.zeros(k_kc, dtype=torch.bool, device=dev)
    kc_b = torch.zeros(k_kc, dtype=torch.bool, device=dev)
    for name, odor, store in (("A", A, kc_a), ("B", Bo, kc_b)):
        c = play(1.0, odor, record=True)
        play(2.0, None)
        store |= (c[:k_kc][:, torch.tensor(group_of == groups.index("control"), device=dev)].sum(1) > 0)
    only_a, only_b = kc_a & ~kc_b, kc_b & ~kc_a
    syn_a, syn_b = only_a[mb.kc_row], only_b[mb.kc_row]
    print(f"KCs recruited: A {int(kc_a.sum())}, B {int(kc_b.sum())}, both {int((kc_a & kc_b).sum())}; "
          f"plastic synapses from A-only KCs {int(syn_a.sum())}, B-only {int(syn_b.sum())}")
    m_before = mb.summary()
    for k in range(args.trainings):
        play(0.5, A, dan=False)                 # odor first, dopamine in its second half (forward pairing)
        play(0.5, A, dan=True)
        play(3.0, None)
    m_after = mb.summary()
    post = test("post")
    if cuda:
        torch.cuda.synchronize()
    # memory per group and MBON type
    mem = {}
    for g, gname in enumerate(groups):
        cols = torch.tensor(np.flatnonzero(group_of == g), device=dev)
        mem[gname] = mb.summary(cols)
    spec = {}
    for g, gname in enumerate(groups):
        cols = torch.tensor(np.flatnonzero(group_of == g), device=dev)
        mm = brain.m[:, cols]
        spec[gname] = {"depressed_frac": float((mm < 0.9).float().mean()), "m_syn_from_A_only_KCs": float(mm[syn_a].mean()),
                       "m_syn_from_B_only_KCs": float(mm[syn_b].mean()), "m_all": float(mm.mean())}
    print("\nmemory specificity (mean multiplier of synapses from KCs that only A / only B recruited; fraction < 0.9):")
    for gname in groups:
        r = spec[gname]
        print(f"  {gname:10s} A-only {r['m_syn_from_A_only_KCs']:.3f}  B-only {r['m_syn_from_B_only_KCs']:.3f}  "
              f"all {r['m_all']:.3f}  depressed {100 * r['depressed_frac']:.1f}%")
    print("\nmultipliers after training (1 = unchanged), MBON types that changed most:")
    for gname in groups:
        changed = sorted(mem[gname], key=lambda t: mem[gname][t])[:6]
        print(f"  {gname:10s} " + ", ".join(f"{t} {mem[gname][t]:.2f}" for t in changed))
    print("\nMBON response to A vs B, pre -> post (Hz):")
    for gname in groups:
        print(f"  {gname:10s} A {pre['A'][gname]['mbon_mean_hz']:.2f} -> {post['A'][gname]['mbon_mean_hz']:.2f}   "
              f"B {pre['B'][gname]['mbon_mean_hz']:.2f} -> {post['B'][gname]['mbon_mean_hz']:.2f}")
    print("\nDN rates to A, pre -> post:")
    for key in pre["A"]["control"]["dn"]:
        print(f"  {key:10s} " + "  ".join(f"{g[:4]} {pre['A'][g]['dn'][key]:5.1f}->{post['A'][g]['dn'][key]:5.1f}" for g in groups))
    out = {"args": vars(args), "pre": pre, "post": post, "memory": mem, "specificity": spec,
           "overflow": brain.overflow_kind.cpu().numpy().tolist(),
           "seconds": round(time.time() - t0, 1)}
    path = config.RESULTS / f"mb_learning_check_eta{args.eta:g}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"overflow {out['overflow']}, {out['seconds']} s, wrote {path}")


if __name__ == "__main__":
    main()
