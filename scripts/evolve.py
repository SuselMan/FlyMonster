"""Evolve Eye + Nose fly pairs through the maze curriculum.

Evolution strategy (antithetic sampling, rank fitness, Adam) over the wiring
parameters; brains themselves are the fixed connectome. Every generation all
pairs play the same fresh maps. The unperturbed "center" pair is also played
with hearing muted, to measure whether messages matter.

Outputs in results/<run>/:
  log.jsonl         one line per generation
  milestones.jsonl  first food, success thresholds, level ups, ...
  replays/gen_XXXXX.json  center pair episodes of that generation
  replays/milestone_*.json
  checkpoint.pt
Usage: python scripts/evolve.py --run first [--gens 500] [--resume]
"""
import argparse
import json
import math
import sys
import time
from collections import Counter, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from flysim import body, config, connectome, world  # noqa: E402
from flysim.team import LEVELS, Policy, Runner  # noqa: E402

SHOWCASE = [777000, 888000]  # + level index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="first")
    ap.add_argument("--gens", type=int, default=500)
    ap.add_argument("--half-pop", type=int, default=8, help="antithetic pairs; population = 2x")
    ap.add_argument("--maps", type=int, default=4, help="maps per generation")
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    out = config.RESULTS / args.run
    (out / "replays").mkdir(parents=True, exist_ok=True)
    con = connectome.load()
    wiring = body.build(con, seed=args.seed)
    runner = Runner(con, wiring)
    policy = Policy(wiring.n_pools)
    rng = np.random.default_rng(args.seed)

    state = {"theta": policy.init(rng), "m": np.zeros(policy.size), "v": np.zeros(policy.size),
             "gen": 0, "level": args.level, "milestones": [], "history": []}
    ckpt = out / "checkpoint.pt"
    if args.resume and ckpt.exists():
        state = torch.load(ckpt, weights_only=False)
        # Checkpoints from before multi-symbol messages have no stored layout.
        old_shapes = state.get("shapes") or Policy(wiring.n_pools, slots=1).shapes
        if old_shapes != policy.shapes:
            fresh = policy.init(rng)
            state["theta"] = policy.migrate(state["theta"], old_shapes, fresh)
            state["m"] = policy.migrate(state["m"], old_shapes, np.zeros(policy.size))
            state["v"] = policy.migrate(state["v"], old_shapes, np.zeros(policy.size))
            print(f"migrated parameters: {sum(int(np.prod(s)) for s in old_shapes.values())} -> {policy.size}")
        print(f"resumed at generation {state['gen']}, level {state['level']}")
    else:
        for f in ("log.jsonl", "milestones.jsonl"):
            (out / f).unlink(missing_ok=True)
    state["shapes"] = policy.shapes
    recent = deque(state["history"][-5:], maxlen=5)

    def milestone(key, text, episode=None):
        if key in state["milestones"]:
            return
        state["milestones"].append(key)
        rec = {"gen": state["gen"], "level": state["level"], "key": key, "text": text}
        if episode is not None:
            rec["replay"] = f"milestone_{key}.json"
            write_json(out / "replays" / rec["replay"], {"gen": state["gen"], "episodes": [episode]})
        append_jsonl(out / "milestones.jsonl", rec)
        print(f"  ★ {text}")

    while state["gen"] < args.gens:
        gen, level = state["gen"], LEVELS[state["level"]]
        t0 = time.perf_counter()
        half = args.half_pop
        # Only parameters used at this level are explored (e.g. no 2nd symbol before level 3).
        noise = rng.normal(0, 1, (half, policy.size)) * policy.active_mask(level)
        members = np.concatenate([state["theta"] + args.sigma * noise,
                                  state["theta"] - args.sigma * noise,
                                  state["theta"][None], state["theta"][None]])
        n_members = len(members)            # 2*half perturbed, center, center muted
        center, muted = 2 * half, 2 * half + 1
        seeds = [gen * 1000 + j for j in range(args.maps)]

        # Plus the center pair on fixed showcase maps, to compare generations on the same maze.
        theta = torch.tensor(np.concatenate([np.repeat(members, args.maps, axis=0),
                                             np.repeat(state["theta"][None], len(SHOWCASE), axis=0)]),
                             dtype=torch.float32)
        mute = torch.zeros(len(theta), dtype=torch.bool)
        mute[muted * args.maps:n_members * args.maps] = True
        showcase = [s + state["level"] for s in SHOWCASE]
        eps = runner.run(theta, seeds * n_members + showcase, level, mute_hearing=mute)
        show_eps = eps[n_members * args.maps:]
        by_member = [eps[i * args.maps:(i + 1) * args.maps] for i in range(n_members)]

        fit = np.array([np.mean([e.fitness(level.max_ticks) for e in m]) for m in by_member])
        # Rank-based fitness, antithetic gradient, Adam.
        ranks = np.empty(2 * half)
        ranks[np.argsort(fit[:2 * half])] = np.linspace(-0.5, 0.5, 2 * half)
        grad = ((ranks[:half] - ranks[half:])[:, None] * noise).sum(0) / (2 * half * args.sigma)
        state["m"] = 0.9 * state["m"] + 0.1 * grad
        state["v"] = 0.999 * state["v"] + 0.001 * grad ** 2
        t = gen + 1
        step = args.lr * (state["m"] / (1 - 0.9 ** t)) / (np.sqrt(state["v"] / (1 - 0.999 ** t)) + 1e-8)
        state["theta"] = state["theta"] + step

        c_eps, m_eps = by_member[center], by_member[muted]
        rate = lambda es, what: float(np.mean([e.outcome == what for e in es]))
        rec = {
            "gen": gen, "level": state["level"], "level_name": level.name,
            "seconds": round(time.perf_counter() - t0, 1),
            "success": rate(c_eps, "food"), "trap": rate(c_eps, "trap"),
            "fitness": float(fit[center]), "pop_best": float(fit[:2 * half].max()),
            "pop_mean": float(fit[:2 * half].mean()),
            "pop_success": float(np.mean([e.outcome == "food" for m in by_member[:2 * half] for e in m])),
            "ticks": float(np.mean([e.ticks for e in c_eps])),
        }
        if level.team:
            rec["muted_success"] = rate(m_eps, "food")
            rec.update(language_stats(c_eps))
        append_jsonl(out / "log.jsonl", rec)
        write_json(out / "replays" / f"gen_{gen:05d}.json",
                   {"gen": gen, "level": state["level"], "level_name": level.name, "team": level.team,
                    "episodes": [episode_json(e) for e in c_eps],
                    "showcase": [episode_json(e) for e in show_eps]})
        write_json(out / "status.json", {"gen": gen, "level": state["level"], "level_name": level.name,
                                         "seconds": rec["seconds"], "time": time.time(), "gens": args.gens})

        comm = (f" muted {rec['muted_success']:.0%} MI(sym;trap) {rec['mi_trap']:.2f}"
                if level.team else "")
        print(f"gen {gen:4d} L{state['level'] + 1} {rec['seconds']:5.1f}s  success {rec['success']:.0%} "
              f"trap {rec['trap']:.0%}  pop success {rec['pop_success']:.0%}  "
              f"fit {rec['fitness']:+.2f} best {rec['pop_best']:+.2f}{comm}")

        # Milestones.
        any_food = [e for m in by_member for e in m if e.outcome == "food"]
        if any_food:
            milestone(f"L{state['level'] + 1}_first_food_any",
                      f"level {state['level'] + 1}: some fly reached food for the first time",
                      episode_json(any_food[0]))
        if rec["success"] > 0:
            milestone(f"L{state['level'] + 1}_center_food", f"level {state['level'] + 1}: main pair reached food",
                      episode_json(next(e for e in c_eps if e.outcome == "food")))
        recent.append(rec["success"])
        state["history"].append(rec["success"])
        avg = float(np.mean(recent))
        for thr in (0.25, 0.5, 0.75):
            if len(recent) == recent.maxlen and avg >= thr:
                milestone(f"L{state['level'] + 1}_success_{int(thr * 100)}",
                          f"level {state['level'] + 1}: success {int(thr * 100)}% over 5 generations")
        if level.team and len(recent) == recent.maxlen:
            logs = read_jsonl(out / "log.jsonl")[-5:]
            gap = np.mean([r["success"] - r.get("muted_success", r["success"]) for r in logs])
            if gap >= 0.2:
                milestone(f"L{state['level'] + 1}_comm_matters",
                          f"level {state['level'] + 1}: muting messages drops success by {gap:.0%}")

        state["gen"] += 1
        if len(recent) == recent.maxlen and avg >= 0.7 and state["level"] + 1 < len(LEVELS):
            state["level"] += 1
            recent.clear()
            milestone(f"level_{state['level'] + 1}", f"level up: {LEVELS[state['level']].name}")
        torch.save(state, ckpt)


def language_stats(episodes) -> dict:
    """How much the Eye's message tells about the situation (mutual information, bits).

    mi_trap / mi_move are for the first symbol; with two-symbol messages also
    for the second symbol (_2) and for the whole pair (_pair).
    """
    rows = []  # (symbols tuple, trap ahead, best move)
    for e in episodes:
        safe = world.bfs(e.map.grid, e.map.food, blocked=e.map.traps)
        for pos, heading, _, _, sym_eye, _ in e.trace:
            fwd = world.vision(e.map, pos, heading)[0]
            rows.append((sym_eye, fwd.trap_dist is not None and fwd.trap_dist <= 2,
                         best_move(e.map, safe, pos, heading)))
    if not rows:
        return {"mi_trap": 0.0, "mi_move": 0.0, "symbols": {}}
    out = {"mi_trap": mutual_info([(s[0], t) for s, t, _ in rows]),
           "mi_move": mutual_info([(s[0], m) for s, _, m in rows]),
           "symbols": {"".join(map(str, k)): v for k, v in sorted(Counter(s for s, _, _ in rows).items())}}
    if len(rows[0][0]) > 1:
        out["mi_trap_2"] = mutual_info([(s[1], t) for s, t, _ in rows])
        out["mi_move_2"] = mutual_info([(s[1], m) for s, _, m in rows])
        out["mi_trap_pair"] = mutual_info([(s, t) for s, t, _ in rows])
        out["mi_move_pair"] = mutual_info([(s, m) for s, _, m in rows])
    return out


def best_move(m, safe_dist, pos, heading) -> str:
    """Relative direction of the next tile on the safe shortest path."""
    here = safe_dist.get(pos)
    if here is None:
        return "none"
    for turn, name in ((0, "fwd"), (-1, "left"), (1, "right"), (2, "back")):
        dr, dc = world.DIRS[(heading + turn) % 4]
        if safe_dist.get((pos[0] + dr, pos[1] + dc), math.inf) < here:
            return name
    return "none"


def mutual_info(pairs) -> float:
    if not pairs:
        return 0.0
    n = len(pairs)
    joint, xs, ys = Counter(pairs), Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    return float(sum(c / n * math.log2(c * n / (xs[x] * ys[y])) for (x, y), c in joint.items()))


def episode_json(e) -> dict:
    safe = world.bfs(e.map.grid, e.map.food, blocked=e.map.traps)
    trace = []
    slot = lambda syms, i: syms[i] if syms is not None and len(syms) > i else None
    for p, h, a, sn, se, ev in e.trace:
        fwd = world.vision(e.map, p, h)[0]
        trace.append({"pos": p, "heading": h, "action": a,
                      "sym_nose": slot(sn, 0), "sym_nose2": slot(sn, 1),
                      "sym_eye": slot(se, 0), "sym_eye2": slot(se, 1), "event": ev,
                      "trap_ahead": fwd.trap_dist is not None and fwd.trap_dist <= 2,
                      "best_move": best_move(e.map, safe, p, h)})
    return {
        "seed": e.map.seed, "outcome": e.outcome, "ticks": e.ticks,
        "grid": e.map.grid.tolist(), "food": e.map.food, "traps": sorted(e.map.traps),
        "trace": trace,
    }


def write_json(path, obj):
    path.write_text(json.dumps(obj, default=int))


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=int) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


if __name__ == "__main__":
    main()
