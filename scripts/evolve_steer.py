"""Evolve the missing ventral nerve cord: how descending-neuron activity turns the body.

FAFB/FlyWire is brain only; the ventral nerve cord that turns descending-neuron
(DN) commands into leg movements is not in the data, so every body mapping is a
guess. Here the brain stays the untouched connectome model, and only the mapping
evolves: turn rate = sum_k w_k * (L - R rate of DN type k, high-passed like the
life body) + b. Candidate DN types are those that carried odor or wind side in
scripts/dn_scan.py and scripts/wind_scan.py, plus DNa01/DNa02.

Task per trial: open 400 x 400 mm ground, one apple (fruit odor plume stretched
downwind, visible as an object), two odorless stones as visual distractors, a
steady wind. Flies start 80-130 mm away (most in the downwind sector, where the
plume reaches) and walk at a fixed speed with the life-body wander noise.
Fitness: reaching the apple (8 mm) early scores 1..2; otherwise the fraction of
the start distance closed (max 0.8). Two controls ride along every generation:
zero weights (pure wander) and the life mapping (DNa01 + DNa02).

Inputs to each brain: vision salience (the life VPN groups), fruit olfaction,
wind on the antennae and the E-PG compass, constant hunger bias.

Results -> results/evolve_steer/: log.jsonl (per generation), best.json, population.npz (resume).
Usage: python scripts/evolve_steer.py [--pop 64] [--trial-s 40] [--trials 2] [--hours 8] [--resume]
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
from flysim.senses import Compass, Olfaction, Wind  # noqa: E402

CANDIDATES = ["DNa01", "DNa02", "DNb05", "DNp18", "DNg99", "DNb06", "DNbe001", "DNp33", "DNg13", "DNp35",
              "DNg56", "DNp06", "DNp31", "DNa04", "DNa10", "DNp19", "DNg96", "DNg31", "DNp73", "DNa11"]
DT = 0.010              # s, world step = one brain block
READ_TAU = 0.05         # s, readout low-pass (life body)
ADAPT_TAU = 5.0         # s, body adapts to a sustained left-right difference (life body)
WALK = 14.0             # mm/s
WANDER = 1.2            # rad/s^0.5
SIZE = 400.0
REACH = 8.0
ODOR_SIGMA = 30.0
HUNGER = 2.0
W_SCALE = 1 / 50.0      # weights act on Hz / 50
TURN_MAX = 4.0          # rad/s


class _Wiring:
    _wire = Life._wire

    def __init__(self, con, meta, dev):
        self.con, self.meta, self.dev = con, meta, dev
        self._wire()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--trial-s", type=float, default=40.0)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--elite", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    rng = np.random.default_rng(args.seed)
    out = config.RESULTS / "evolve_steer"
    out.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda")
    con = connectome.load()
    meta = neuron_meta(con)
    model = apply(con, DEFAULT, meta)
    params = DEFAULT.lif()
    ct = meta.cell_type.fillna("").to_numpy()
    side = meta.side.fillna("").to_numpy()
    w = _Wiring(con, meta, dev)
    B = args.pop
    K = len(CANDIDATES)

    # readout: every cell of the candidate types; mixing matrix -> (K, 2) mean L / R rate
    read_idx, rows = [], []
    for k, ty in enumerate(CANDIDATES):
        for s_i, s in enumerate(("left", "right")):
            cells = np.flatnonzero((ct == ty) & (side == s))
            read_idx.extend(cells.tolist())
            rows.extend([(k, s_i)] * len(cells))
    M = torch.zeros(K * 2, len(read_idx), device=dev)
    for j, (k, s_i) in enumerate(rows):
        M[k * 2 + s_i, j] = 1
    M /= M.sum(1, keepdim=True).clamp(min=1)
    missing = [ty for k, ty in enumerate(CANDIDATES) if M[k * 2].sum() == 0 or M[k * 2 + 1].sum() == 0]
    print(f"{len(read_idx)} readout cells; candidate types without both sides: {missing}")

    olf = Olfaction(meta, B, dev)
    wind = Wind(meta, dev)
    compass = Compass(con, meta, dev)
    input_idx = torch.cat([w.sense_idx, olf.input_idx, wind.input_idx, compass.input_idx])
    n_s, n_o, n_w = len(w.sense_idx), len(olf.input_idx), len(wind.input_idx)
    brain = FastBrain(model, B, params, input_idx, torch.tensor(read_idx, device=dev), w.npf_idx,
                      steps=round(DT * 1000 / params.dt), max_spikes=256 * B, max_events=40_000 * B)
    brain.bias.fill_(HUNGER)
    gi = {n: i for i, n in enumerate(w.group_names)}
    fruit = olf.names.index("fruit")

    # population: weights (K) + bias; slot 0 = zero control, slot 1 = life mapping control
    ctrl_life = np.zeros(K + 1)
    ctrl_life[CANDIDATES.index("DNa01")] = ctrl_life[CANDIDATES.index("DNa02")] = 0.025 * 50   # life turn_gain
    gen0 = 0
    if args.resume and (out / "population.npz").exists():
        z = np.load(out / "population.npz")
        pop, gen0 = z["pop"], int(z["gen"]) + 1
        print(f"resumed at generation {gen0}")
    else:
        pop = np.zeros((B - 2, K + 1))
        pop[:, :K] = rng.normal(0, 0.5, (B - 2, K)) * (rng.random((B - 2, K)) < 0.4)
        pop[:len(pop) // 4] += ctrl_life                      # a quarter starts near the life mapping
    t_end = time.time() + args.hours * 3600

    gen = gen0
    while time.time() < t_end:
        genomes = np.vstack([np.zeros(K + 1), ctrl_life, pop])
        t0 = time.perf_counter()
        fit = np.zeros(B)
        reached = np.zeros(B)
        for trial in range(args.trials):
            f, r = run_trial(brain, olf, wind, compass, M, genomes, rng, args.trial_s, gi, fruit, n_s, n_o, n_w, w)
            fit += f / args.trials
            reached += r
        wall = time.perf_counter() - t0
        order = np.argsort(-fit[2:])
        best = pop[order[0]]
        rec = {"gen": gen, "wall_s": round(wall, 1), "best": round(float(fit[2:][order[0]]), 3),
               "mean": round(float(fit[2:].mean()), 3), "top10_mean": round(float(fit[2:][order[:10]].mean()), 3),
               "ctrl_zero": round(float(fit[0]), 3), "ctrl_life": round(float(fit[1]), 3),
               "reached_pop": int(reached[2:].sum()), "reached_ctrl_zero": int(reached[0]), "reached_ctrl_life": int(reached[1]),
               "trials": args.trials * (B - 2),
               "best_genome": {n: round(float(v), 3) for n, v in zip(CANDIDATES + ["bias"], best)}}
        with open(out / "log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"gen {gen:3d}  {wall:6.1f}s  best {rec['best']:.2f}  top10 {rec['top10_mean']:.2f}  mean {rec['mean']:.2f}  "
              f"zero {rec['ctrl_zero']:.2f}  life {rec['ctrl_life']:.2f}  reached {rec['reached_pop']}/{rec['trials']}", flush=True)
        # next generation: elites stay, the rest from tournament parents, uniform crossover, mutation
        elite = pop[order[:args.elite]]
        kids = []
        while len(kids) < len(pop) - args.elite:
            a, b = (pop[min(rng.integers(0, len(pop), 3), key=lambda i: -fit[2:][i])] for _ in range(2))
            child = np.where(rng.random(K + 1) < 0.5, a, b)
            child = child + rng.normal(0, args.sigma, K + 1) * (rng.random(K + 1) < 0.3)
            child[rng.random(K + 1) < 0.03] = 0.0
            kids.append(child)
        pop = np.vstack([elite, kids])
        np.savez(out / "population.npz", pop=pop, gen=gen)
        (out / "best.json").write_text(json.dumps({"gen": gen, "fitness": rec["best"], "candidates": CANDIDATES,
                                                   "genome": best.tolist(), "w_scale": W_SCALE, "adapt_tau": ADAPT_TAU}))
        gen += 1


@torch.no_grad()
def run_trial(brain, olf, wind, compass, M, genomes, rng, trial_s, gi, fruit, n_s, n_o, n_w, w):
    B, K = len(genomes), M.shape[0] // 2
    dev = brain.dev
    # world: apple near the middle, wind, two stones; flies start 80-130 mm away
    fx, fy = rng.uniform(150, 250, 2)
    wang, wspd = rng.uniform(-np.pi, np.pi), rng.uniform(0.3, 0.8)
    ux, uy = np.cos(wang), np.sin(wang)                        # direction the air flows to
    stones = rng.uniform(60, 340, (2, 2))
    d0 = rng.uniform(80, 130, B)
    downwind = rng.random(B) < 0.7
    a0 = np.where(downwind, wang + rng.uniform(-1.0, 1.0, B), rng.uniform(-np.pi, np.pi, B))
    x, y = np.clip(fx + d0 * np.cos(a0), 5, SIZE - 5), np.clip(fy + d0 * np.sin(a0), 5, SIZE - 5)
    d0 = np.hypot(x - fx, y - fy)
    h = rng.uniform(-np.pi, np.pi, B)
    read = torch.zeros(2 * K, B, device=dev)
    base = torch.zeros(K, B, device=dev)
    Wt = torch.tensor(genomes[:, :K], dtype=torch.float32, device=dev)
    bias = genomes[:, K]
    brain.reset_all()
    olf.state.zero_()
    dmin, t_reach = d0.copy(), np.full(B, np.inf)
    steps = round(trial_s / DT)
    wind_speed = torch.full((B,), wspd, device=dev)
    for k in range(steps):
        rates = torch.zeros(len(brain.input_idx), B, device=dev)
        # vision: apple (r 6 mm) and stones (r 12 mm) as objects in each hemifield (life formula)
        drive = np.zeros((len(w.group_names), B), dtype=np.float32)
        ox, oy, osz = np.array([fx, *stones[:, 0]]), np.array([fy, *stones[:, 1]]), np.array([6.0, 12.0, 12.0])
        dx, dy = ox[None] - x[:, None], oy[None] - y[:, None]
        dist = np.hypot(dx, dy) + 1e-6
        az = np.angle(np.exp(1j * (np.arctan2(dy, dx) - h[:, None])))
        ang = np.clip(2 * osz[None] / dist, 0, 1.5) * (dist < 80)
        drive[gi["vis_L"]] = 150 * np.clip((ang * np.clip(az / 0.5, 0, 1)).sum(1) / 0.5, 0, 1)
        drive[gi["vis_R"]] = 150 * np.clip((ang * np.clip(-az / 0.5, 0, 1)).sum(1) / 0.5, 0, 1)
        rates[:n_s] = torch.from_numpy(drive).to(dev)[w.sense_col]
        # odor at the two antennae (arena plume formula)
        conc = np.zeros((B, len(olf.names), 2), dtype=np.float32)
        for s_i, sign in ((0, 1), (1, -1)):
            ax = x + 1.2 * np.cos(h) - sign * 0.8 * np.sin(h)
            ay = y + 1.2 * np.sin(h) + sign * 0.8 * np.cos(h)
            ddx, ddy = ax - fx, ay - fy
            along, cross = ddx * ux + ddy * uy, -ddx * uy + ddy * ux
            s_al = np.where(along > 0, ODOR_SIGMA * (1 + 2.5 * wspd), ODOR_SIGMA)
            conc[:, fruit, s_i] = np.exp(-along ** 2 / (2 * s_al ** 2) - cross ** 2 / (2 * ODOR_SIGMA ** 2))
        rates[n_s:n_s + n_o] = olf.rates(torch.from_numpy(conc).to(dev), DT * 1000)
        rel = np.angle(np.exp(1j * (np.arctan2(-uy, -ux) - h)))
        rates[n_s + n_o:n_s + n_o + n_w] = wind.rates(wind_speed, torch.tensor(rel, dtype=torch.float32, device=dev))
        rates[n_s + n_o + n_w:] = compass.rates(torch.tensor(h, dtype=torch.float32, device=dev))
        brain.rates.copy_(rates)
        brain.run()
        read += (M @ brain.counts / DT - read) * (DT / READ_TAU)
        diff = read[0::2] - read[1::2]                                  # (K, B) left - right
        base += (diff - base) * (DT / ADAPT_TAU)
        turn = ((Wt.T * (diff - base)).sum(0) * W_SCALE).cpu().numpy() + bias
        turn = np.clip(turn, -TURN_MAX, TURN_MAX)
        h = h + turn * DT + WANDER * np.sqrt(DT) * rng.normal(0, 1, B)
        x, y = x + WALK * np.cos(h) * DT, y + WALK * np.sin(h) * DT
        hit_x, hit_y = (x < 2) | (x > SIZE - 2), (y < 2) | (y > SIZE - 2)
        h = np.where(hit_x, np.pi - h, h)
        h = np.where(hit_y, -h, h)
        x, y = np.clip(x, 2, SIZE - 2), np.clip(y, 2, SIZE - 2)
        d = np.hypot(x - fx, y - fy)
        dmin = np.minimum(dmin, d)
        t_reach = np.where((d < REACH) & np.isinf(t_reach), k * DT, t_reach)
    if float(brain.overflow) > 0:
        print("warning: brain buffer overflow in this trial", flush=True)
        brain.max_spikes, brain.max_events = brain.max_spikes * 2, brain.max_events * 2
        brain._alloc_events()
        brain.overflow.zero_()
        brain.capture()
    reached = np.isfinite(t_reach)
    fit = np.where(reached, 1 + (trial_s - t_reach) / trial_s, 0.8 * np.clip((d0 - dmin) / d0, 0, 1))
    return fit, reached.astype(float)


if __name__ == "__main__":
    main()
