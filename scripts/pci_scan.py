"""Perturbational complexity index (PCI) of the fly brain model.

Casali et al. 2013 (Sci Transl Med): perturb the brain, record the evoked response over many
sources, binarize it against the baseline and take the Lempel-Ziv complexity of the binary
space x time matrix, normalised by its entropy. In humans PCI separates wakefulness (0.4-0.7)
from anaesthesia, deep sleep and coma (< 0.3): a conscious brain answers a perturbation with a
response that is both widespread and differentiated; an unconscious one with silence or a
stereotyped wave.

Here:
- sources ("channels") are FlyWire cell types, neurons of a type pooled; directly stimulated
  neurons are excluded;
- the perturbation is a brief Poisson pulse to a neuron set: a random set of central-brain
  neurons (like TMS), Johnston's organ, or visual projection neurons;
- background: all sensory neurons fire at a low Poisson rate, so the brain has ongoing activity;
- "states" are physiological conditions of the same connectome:
    awake        default physiology
    cold<k>      all synaptic weights x k (chill coma analog: reduced excitability)
    thr<d>       spike threshold raised by d mV
    shuffled     synapse targets permuted (same out-degrees, no structure): a control that
                 should score low if the measure means anything
- sham trials (background only) give the baseline; the significance threshold per channel is a
  bootstrap of the maximum over time (family-wise alpha), as in Casali et al.

Writes results/pci_scan.json.
Usage: python scripts/pci_scan.py [--stim random|jon|vision] [--conditions awake,cold0.5,cold0.25,shuffled]
                                  [--reps 8] [--post 300] [--device cuda]   (--selftest checks lz76)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from flysim import config, connectome, neurons  # noqa: E402
from flysim.fastbrain import FastBrain  # noqa: E402
from flysim.physiology import DEFAULT, apply, neuron_meta  # noqa: E402


# --- Lempel-Ziv complexity ----------------------------------------------------------------------
def lz76_reference(s: bytes) -> int:
    """Kaspar & Schuster (1987) LZ76 phrase count. Quadratic on sparse data; kept for --selftest."""
    n = len(s)
    if n < 2:
        return n
    i, k, l, c, k_max = 0, 1, 1, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c


def _suffix_array(s: np.ndarray):
    """Prefix doubling. Returns (sa, ranks): ranks[j][i] = rank of suffix i by its first 2**j symbols."""
    n = len(s)
    rank = s.astype(np.int64)
    ranks = [rank.astype(np.int32)]
    k = 1
    while True:
        r2 = np.full(n, -1, np.int64)
        if k < n:
            r2[:n - k] = rank[k:]
        key = rank * (n + 1) + (r2 + 1)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        new = np.concatenate([[0], np.cumsum(ks[1:] != ks[:-1])])
        rank = np.empty(n, np.int64)
        rank[order] = new
        ranks.append(rank.astype(np.int32))
        if new[-1] == n - 1 or k >= n:
            return order, ranks
        k *= 2


def lz76(bits: np.ndarray) -> int:
    """LZ76 phrase count (same parsing as Kaspar & Schuster) in near-linear time: suffix array,
    LCP by binary lifting, longest previous factor by the stack algorithm of Crochemore & Ilie
    (2008); each phrase is the longest previous factor plus one symbol."""
    s = np.asarray(bits, dtype=np.uint8).reshape(-1)
    n = len(s)
    if n < 2:
        return n
    sa, ranks = _suffix_array(s)
    a, b = sa[:-1], sa[1:]
    lcp = np.zeros(n - 1, np.int64)                  # lcp of neighbours in suffix order
    for j in range(len(ranks) - 1, -1, -1):
        step = 1 << j
        ok = (a + lcp + step <= n) & (b + lcp + step <= n)
        ia, ib = np.minimum(a + lcp, n - 1), np.minimum(b + lcp, n - 1)
        lcp += (ok & (ranks[j][ia] == ranks[j][ib])) * step
    SA = sa.tolist() + [-1]
    LCP = [0] + lcp.tolist() + [0]
    lpf = [0] * n
    stack = [0]
    for i in range(1, n + 1):
        sai, lci = SA[i], LCP[i]
        while stack:
            t = stack[-1]
            sat, lct = SA[t], LCP[t]
            if sai < sat:
                lpf[sat] = lct if lct > lci else lci
                if lct < lci:
                    lci = lct
                stack.pop()
            elif lci <= lct:
                lpf[sat] = lct
                stack.pop()
            else:
                break
        LCP[i] = lci
        if i < n:
            stack.append(i)
    l, c = 0, 0
    while l < n:
        l += lpf[l] + 1
        c += 1
    return c


def selftest(trials: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        bits = (rng.random(int(rng.integers(1, 300))) < rng.choice([0.5, 0.1, 0.02])).astype(np.uint8)
        assert lz76(bits) == lz76_reference(bits.tobytes()), bits.tolist()
    print(f"lz76 matches the reference on {trials} random sequences")


def pci(ss: np.ndarray) -> dict:
    """PCI of a binary (channels, time) matrix: rows sorted by activity, scanned time-major."""
    ss = ss.astype(bool)
    p = ss.mean()
    if p == 0 or p == 1:
        return {"pci": 0.0, "lz": 0, "p": float(p), "L": int(ss.size)}
    order = np.argsort(-ss.sum(1), kind="stable")
    seq = ss[order].T.reshape(-1)                      # time-major: for each bin, all channels
    L = seq.size
    c = lz76(seq)
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))   # source entropy
    return {"pci": float(c * np.log2(L) / (L * h)), "lz": int(c), "p": float(p), "L": int(L)}


def binarize(pert: np.ndarray, sham: np.ndarray, n_pre: int, alpha: float = 0.01, boots: int = 200,
             seed: int = 0) -> np.ndarray:
    """Significant post-stimulus response per channel and bin.

    pert, sham: (reps, T, C) spike counts per bin. The null of a channel is its trial-averaged
    count in bins without stimulus (all sham bins + pre-stimulus perturbed bins); the threshold is
    the (1 - alpha) quantile of the bootstrapped maximum over the post-stimulus bins.
    """
    P = pert.mean(0)                                   # (T, C)
    S = sham.mean(0)
    null = np.concatenate([S, P[:n_pre]], 0)           # (N, C)
    post = P[n_pre:]                                   # (T_post, C)
    rng = np.random.default_rng(seed)
    T_post, C = post.shape
    draws = rng.integers(0, len(null), size=(boots, T_post))
    maxes = null[draws].max(1)                         # (boots, C)
    thr = np.quantile(maxes, 1 - alpha, axis=0)
    return (post > thr).T                              # (C, T_post)


# --- model variants ---------------------------------------------------------------------------
def scaled(con: connectome.Connectome, k: float) -> connectome.Connectome:
    w = con.weights
    return connectome.Connectome(con.ids, torch.sparse_csr_tensor(w.crow_indices(), w.col_indices(), w.values() * k,
                                                                  w.shape), con.index_of)


def shuffled(con: connectome.Connectome, seed: int) -> connectome.Connectome:
    """Permute synapse targets across the whole connectome: out-degree and weights per presynaptic
    neuron are kept, the wiring diagram is destroyed."""
    w = con.weights
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(w.col_indices().numel(), generator=g)
    return connectome.Connectome(con.ids, torch.sparse_csr_tensor(w.crow_indices(), w.col_indices()[perm], w.values(),
                                                                  w.shape), con.index_of)


def condition(name: str, model, params, seed: int):
    if name == "awake":
        return model, params
    if name.startswith("cold"):
        return scaled(model, float(name[4:])), params
    if name.startswith("thr"):
        p = config.LIFParams(**{**params.__dict__, "v_th": params.v_th + float(name[3:])})
        return model, p
    if name == "shuffled":
        return shuffled(model, seed), params
    raise SystemExit(f"unknown condition {name!r}")


# --- main -------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim", default="random", choices=["random", "jon", "vision"])
    ap.add_argument("--stim-n", type=int, default=500, help="neurons in the random/vision stimulus set")
    ap.add_argument("--stim-hz", type=float, default=2000.0, help="Poisson rate of the pulse")
    ap.add_argument("--pulse", type=float, default=4.0, help="ms, pulse duration")
    ap.add_argument("--background", type=float, default=5.0, help="Hz, Poisson rate of all sensory neurons")
    ap.add_argument("--warmup", type=float, default=200.0, help="ms, settle before recording")
    ap.add_argument("--pre", type=float, default=100.0, help="ms, recorded before the pulse")
    ap.add_argument("--post", type=float, default=300.0, help="ms, recorded after the pulse")
    ap.add_argument("--bin", type=float, default=2.0, help="ms, time bin (multiple of the synaptic delay)")
    ap.add_argument("--reps", type=int, default=8, help="perturbed trials (and as many sham) per condition")
    ap.add_argument("--conditions", default="awake,cold0.5,cold0.25,shuffled")
    ap.add_argument("--min-neurons", type=int, default=2, help="smallest cell type kept as a channel")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(config.RESULTS / "pci_scan.json"))
    ap.add_argument("--selftest", action="store_true", help="check lz76 against the reference and exit")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    dev = torch.device(args.device)
    cuda = dev.type == "cuda"
    rng = np.random.default_rng(args.seed)

    con = connectome.load()
    meta = neuron_meta(con)
    model = apply(con, DEFAULT, meta)
    params = DEFAULT.lif()
    sup = meta.super_class.fillna("").to_numpy()

    # stimulus and background sets
    if args.stim == "random":
        stim = rng.choice(np.flatnonzero(sup == "central"), args.stim_n, replace=False)
    elif args.stim == "jon":
        stim = np.array([con.index_of[i] for i in neurons.JON_CE if i in con.index_of])
    else:
        stim = rng.choice(np.flatnonzero(sup == "visual_projection"), args.stim_n, replace=False)
    background = np.flatnonzero(np.isin(sup, ["sensory", "sensory_ascending"]))
    input_idx = torch.tensor(np.concatenate([background, stim]), device=dev)
    n_bg = len(background)

    # channels: cell types (hemibrain type as fallback), stimulated neurons excluded
    ctype = meta.cell_type.fillna(meta.hemibrain_type).fillna("").to_numpy().astype(str)
    ctype[stim] = ""
    names, chan_of, sizes = np.unique(ctype, return_inverse=True, return_counts=True)
    keep = (sizes >= args.min_neurons) & (names != "")
    remap = np.full(len(names), -1)
    remap[keep] = np.arange(keep.sum())
    chan_of = remap[chan_of]
    C = int(keep.sum())
    chan_names = names[keep].tolist()
    recorded = np.flatnonzero(chan_of >= 0)
    read_idx = torch.tensor(recorded, device=dev)
    chan_t = torch.tensor(chan_of[recorded], device=dev)

    steps = round(args.bin / params.dt)
    delay = max(1, round(params.t_dly / params.dt))
    if steps % delay:
        raise SystemExit(f"--bin must be a multiple of the synaptic delay ({delay * params.dt} ms)")
    n_warm, n_pre, n_pulse, n_post = (round(x / args.bin) for x in (args.warmup, args.pre, args.pulse, args.post))
    T = n_pre + n_pulse + n_post
    R = args.reps
    B = 2 * R                                           # columns 0..R-1 perturbed, R..2R-1 sham
    conds = args.conditions.split(",")
    print(f"neurons {con.n}, channels {C} ({len(recorded)} neurons), stimulus {args.stim} x{len(stim)}, "
          f"background {n_bg} sensory @ {args.background} Hz, bins {T} x {args.bin} ms, {R}+{R} trials, {dev}")

    results = {"args": vars(args), "channels": C, "conditions": {}}
    for cond in conds:
        t0 = time.time()
        m, p = condition(cond, model, params, args.seed)
        brain = FastBrain(m, B, p, input_idx, read_idx, torch.zeros(0, dtype=torch.long, device=dev),
                          steps=steps, max_spikes=8192 * B, max_events=200_000 * B, device=str(dev),
                          use_kernel=cuda)
        run = brain.run if cuda else brain.run_eager
        rates = torch.zeros(len(input_idx), B, device=dev)
        rates[:n_bg] = args.background
        brain.rates.copy_(rates)
        X = torch.zeros(T, C, B, device=dev)
        torch.manual_seed(args.seed)
        for t in range(-n_warm, T):
            if t == n_pre:
                rates[n_bg:, :R] = args.stim_hz
                brain.rates.copy_(rates)
            if t == n_pre + n_pulse:
                rates[n_bg:, :R] = 0
                brain.rates.copy_(rates)
            run()
            if t >= 0:
                X[t].index_add_(0, chan_t, brain.counts)
        if cuda:
            torch.cuda.synchronize()
        over = brain.overflow_kind.cpu().numpy().tolist()
        x = X.permute(2, 0, 1).cpu().numpy()             # (B, T, C)
        pert, sham = x[:R], x[R:]
        ss = binarize(pert, sham, n_pre, args.alpha, seed=args.seed)
        ss = ss[:, n_pulse:] if n_pulse else ss            # drop the pulse bins themselves
        res = pci(ss)
        active = ss.any(1)
        last = int(np.flatnonzero(ss.any(0)).max() + 1) * args.bin if ss.any() else 0.0
        evoked = float((pert[:, n_pre:].sum((1, 2)) - sham[:, n_pre:].sum((1, 2))).mean())
        bg_hz = float(sham.sum() / (R * T * args.bin * 1e-3) / len(recorded))
        res.update(channels_active=int(active.sum()), response_ms=last, evoked_spikes=evoked,
                   background_hz_per_neuron=bg_hz, overflow=over, seconds=round(time.time() - t0, 1))
        top = np.argsort(-ss.sum(1))[:8]
        res["top_channels"] = [(chan_names[i], int(ss[i].sum())) for i in top if ss[i].any()]
        results["conditions"][cond] = res
        print(f"{cond:10s} PCI {res['pci']:.3f}  LZ {res['lz']:5d}  active channels {res['channels_active']:5d}/{C}"
              f"  response {last:5.0f} ms  evoked spikes {evoked:9.0f}  bg {bg_hz:.2f} Hz/neuron"
              f"  overflow {over}  {res['seconds']}s")
        del brain, X
        if cuda:
            torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
