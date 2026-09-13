"""Flies with whole-brain simulations living in the arena.

Loop every world step (10 ms): world -> sensory rates -> 20 brain steps
(dt 0.5 ms) -> descending-neuron spike counts -> low-pass readout -> body.

What is connectome and what is ours (see README):
- Sensory transduction, the body and physiology (energy, eggs) are our models.
  Between sensory neurons and descending/motor neurons it is the connectome.
- Body mapping uses descending neurons with published roles only:
  DNa01/DNa02 ipsilateral turning, MDN backward walking, DNp01 giant fiber
  takeoff, MN9 proboscis (feeding). Walking itself is an innate generator
  (leg circuits are in the ventral nerve cord, absent from FAFB).
- Hunger lowers the threshold of NPF neurons and raises sugar sensitivity.
- Reproduction is clonal. oviDN do not respond to substrate cues in this model,
  so a mature egg is laid while the fly feeds (feeding is the brain's choice).
- Mushroom-body learning is not included: KC->MBON changes do not move
  descending neurons here (scripts/mb_causal_scan.py).
- The genome holds physiological parameters only (sensory gains, hunger
  sensitivity, walking speed), never preferences for objects.
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from .. import connectome, neurons
from ..body import dn_reach
from ..brain import FlyBrain
from ..physiology import DEFAULT, Physiology, apply, neuron_meta
from ..senses import Olfaction
from .arena import Arena, ArenaConfig

WORLD_DT = 0.010          # s
READ_TAU = 0.050          # s, readout low-pass
READOUT = {               # name -> (cell_type, side or None)
    "DNa01 L": ("DNa01", "left"), "DNa01 R": ("DNa01", "right"),
    "DNa02 L": ("DNa02", "left"), "DNa02 R": ("DNa02", "right"),
    "MDN": ("MDN", None), "GF": ("DNp01", None), "MN9": (None, None),
}
GENES = ("vision", "smell", "taste", "looming", "hunger", "walk")


@dataclass
class LifeConfig:
    n_flies: int = 24
    max_flies: int = 32
    seed: int = 1
    walk_speed: float = 14.0      # mm/s, innate walking generator
    turn_gain: float = 0.025      # rad/s per Hz of left-right DNa01+DNa02 difference
    turn_adapt: float = 5.0       # s, adaptation of the body to a sustained left-right difference
    wander: float = 1.2           # rad/s^0.5, turning noise of the walking generator
    gf_threshold: float = 60.0    # Hz, low-passed giant fiber rate that triggers takeoff
    mdn_threshold: float = 25.0   # Hz, backward walking
    feed_threshold: float = 15.0  # Hz, MN9 while touching food
    metabolism: float = 0.0005    # energy per s at rest (1 = full; ~15 min from full to empty when walking)
    move_cost: float = 0.00004    # energy per mm walked
    jump_cost: float = 0.01
    sugar_per_s: float = 5.0      # food units eaten per s
    energy_per_sugar: float = 0.01
    lifespan: tuple = (2700.0, 3600.0)   # s, 45-60 min of simulated adult life
    maturity: float = 300.0       # s of age before the first egg
    egg_interval: float = 120.0   # s between eggs
    egg_energy: float = 0.2       # energy put into an egg
    hatch_time: float = 90.0      # s from egg to adult (development heavily compressed)
    mutation: float = 0.12        # log-normal sigma of gene mutations
    physiology: Physiology = DEFAULT
    arena: ArenaConfig = field(default_factory=ArenaConfig)


@dataclass
class Egg:
    x: float
    y: float
    laid: float
    parent: int
    generation: int
    genome: np.ndarray


class Life:
    def __init__(self, cfg: LifeConfig, con=None, device=None):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.con = con or connectome.load()
        self.meta = neuron_meta(self.con)
        model = apply(self.con, cfg.physiology, self.meta)
        B = cfg.n_flies
        self.brain = FlyBrain(model, batch=B, params=cfg.physiology.lif(), device=device)
        self.dev = self.brain.device
        self.steps_per_world = round(WORLD_DT * 1000 / self.brain.p.dt)
        self._wire()
        self.olf = Olfaction(self.meta, B, self.dev)
        self.arena = Arena.create(cfg.arena, cfg.seed)
        self.t = 0.0
        self.ids, self.next_id = [], 0
        self.eggs: list[Egg] = []
        self.events: list = []
        self.dead: list = []
        self.births = 0
        self._columns = ("x", "y", "heading", "energy", "age", "lifespan", "airborne", "cooldown", "jumped_at",
                         "state", "turn_base", "last_feed", "last_egg", "generation", "parent", "genome")
        for c in self._columns:
            setattr(self, c, np.zeros((0, len(GENES))) if c == "genome" else np.zeros(0))
        self.read = {k: np.zeros(0) for k in READOUT}
        s = cfg.arena.size
        self._append(self.rng.uniform(20, s - 20, B), self.rng.uniform(20, s - 20, B),
                     np.ones((B, len(GENES))), np.zeros(B), np.full(B, -1), brains_exist=True)

    # --- wiring ---------------------------------------------------------------
    def _wire(self):
        m, con = self.meta, self.con
        ct = m.cell_type.fillna("").to_numpy()
        side = m.side.fillna("").to_numpy()
        sup = m.super_class.fillna("").to_numpy()
        sub = m.cell_sub_class.fillna("").to_numpy()
        reach = dn_reach(con, np.flatnonzero(sup == "descending"))

        def ranked(mask, k):
            idx = np.flatnonzero(mask)
            return idx[np.argsort(-reach[idx], kind="stable")][:k]

        groups = {
            "vis_L": ranked((sup == "visual_projection") & (side == "left"), 60),
            "vis_R": ranked((sup == "visual_projection") & (side == "right"), 60),
            "loom_L": np.flatnonzero(np.isin(ct, ["LC4", "LPLC2"]) & (side == "left")),
            "loom_R": np.flatnonzero(np.isin(ct, ["LC4", "LPLC2"]) & (side == "right")),
            "sugar": np.array(con.indices(i for i in neurons.SUGAR_GRN if i in con.index_of)),
            "touch_L": np.flatnonzero((sub == "head bristle") & (side == "left")),
            "touch_R": np.flatnonzero((sub == "head bristle") & (side == "right")),
        }
        self.group_names = list(groups)
        self.group_gene = {"vis_L": "vision", "vis_R": "vision", "loom_L": "looming", "loom_R": "looming",
                           "sugar": "taste", "touch_L": None, "touch_R": None}
        idx, col = [], []
        for g, name in enumerate(self.group_names):
            idx.extend(groups[name].tolist())
            col.extend([g] * len(groups[name]))
        self.sense_idx = torch.tensor(idx, device=self.dev)
        self.sense_col = torch.tensor(col, device=self.dev)
        # one gather for all readout neurons, then average per readout name
        read_idx, read_row = [], []
        for r, (name, (t, s)) in enumerate(READOUT.items()):
            sel = [con.index_of[neurons.MN9]] if name == "MN9" else \
                np.flatnonzero((ct == t) & ((side == s) if s else True)).tolist()
            read_idx.extend(sel)
            read_row.extend([r] * len(sel))
        self.read_idx = torch.tensor(read_idx, device=self.dev)
        M = torch.zeros(len(READOUT), len(read_idx), device=self.dev)
        M[torch.tensor(read_row), torch.arange(len(read_idx))] = 1
        self.read_mix = M / M.sum(1, keepdim=True)
        self.npf_idx = torch.tensor(np.flatnonzero(np.char.find(ct.astype(str), "NPF") >= 0), device=self.dev)

    # --- population -----------------------------------------------------------
    def _append(self, xs, ys, genomes, generations, parents, brains_exist=False):
        n = len(xs)
        if not brains_exist:
            self.brain.add(n)
            self.olf.add(n)
        new = {
            "x": xs, "y": ys, "heading": self.rng.uniform(-np.pi, np.pi, n), "energy": np.full(n, 0.7),
            "age": np.zeros(n), "lifespan": self.rng.uniform(*self.cfg.lifespan, n), "airborne": np.zeros(n),
            "cooldown": np.zeros(n), "jumped_at": np.full(n, -1e9), "state": np.zeros(n), "turn_base": np.zeros(n),
            "last_feed": np.full(n, -1e9), "last_egg": np.full(n, -1e9), "generation": generations,
            "parent": parents, "genome": genomes,
        }
        for c in self._columns:
            setattr(self, c, np.concatenate([getattr(self, c), new[c]]))
        self.read = {k: np.concatenate([v, np.zeros(n)]) for k, v in self.read.items()}
        born = list(range(self.next_id, self.next_id + n))
        self.ids.extend(born)
        self.next_id += n
        return born

    def _hatch(self):
        ready = [e for e in self.eggs if self.t - e.laid >= self.cfg.hatch_time]
        room = self.cfg.max_flies - len(self.ids)
        for e in ready[:max(room, 0)]:
            self.eggs.remove(e)
            genome = e.genome * np.exp(self.rng.normal(0, self.cfg.mutation, len(GENES)))
            fid = self._append(np.array([e.x]), np.array([e.y]), genome[None], np.array([e.generation]),
                               np.array([e.parent]))[0]
            self.births += 1
            self.events.append({"t": round(self.t, 2), "fly": fid, "kind": "born",
                                "text": f"вылупилась (поколение {e.generation}, мать №{e.parent})"})
        for e in ready[max(room, 0):]:   # no room: the egg does not develop
            if self.t - e.laid > 3 * self.cfg.hatch_time:
                self.eggs.remove(e)

    # --- one world step -----------------------------------------------------
    @torch.no_grad()
    def step(self):
        if not self.ids:
            return
        light = self.arena.light(self.t)
        g = torch.tensor(self.genome, dtype=torch.float32, device=self.dev)     # (B, genes)
        rates = self._sensory_rates(light)
        olf_rates = self._olfaction() * g[:, GENES.index("smell")]
        idx = torch.cat([self.sense_idx, self.olf.input_idx])
        all_rates = torch.cat([rates, olf_rates])
        hunger = torch.tensor(np.clip(1 - self.energy, 0, 1), dtype=torch.float32, device=self.dev)
        npf_drive = 4.0 * hunger * g[:, GENES.index("hunger")] * (self.brain.p.dt / self.brain.p.tau)
        counts = torch.zeros(len(self.read_idx), len(self.ids), device=self.dev)
        for _ in range(self.steps_per_world):
            if len(self.npf_idx):
                self.brain.g[self.npf_idx] += npf_drive
            counts += self.brain.step(idx, all_rates)[self.read_idx]
        hz = ((self.read_mix @ counts) / WORLD_DT).cpu().numpy()
        a = WORLD_DT / READ_TAU
        for r, k in enumerate(READOUT):
            self.read[k] += (hz[r] - self.read[k]) * a
        self._body(light)
        self._hatch()
        self.arena.update(self.t, WORLD_DT, None, self.ids)
        self.t += WORLD_DT

    def _sensory_rates(self, light: float) -> torch.Tensor:
        cfg, B = self.cfg, len(self.ids)
        drive = np.zeros((len(self.group_names), B), dtype=np.float32)
        gi = {n: i for i, n in enumerate(self.group_names)}
        ox = np.concatenate([self.x, [f.x for f in self.arena.food]])
        oy = np.concatenate([self.y, [f.y for f in self.arena.food]])
        osize = np.concatenate([np.full(B, 1.5), [f.radius for f in self.arena.food]])
        dx, dy = ox[None] - self.x[:, None], oy[None] - self.y[:, None]
        dist = np.hypot(dx, dy) + 1e-6
        az = np.angle(np.exp(1j * (np.arctan2(dy, dx) - self.heading[:, None])))   # + = left
        ang = np.clip(2 * osize[None] / dist, 0, 1.5)
        ang[np.arange(B), np.arange(B)] = 0
        ang[dist > 60] = 0
        left = (ang * np.clip(az / 0.5, 0, 1)).sum(1)
        right = (ang * np.clip(-az / 0.5, 0, 1)).sum(1)
        drive[gi["vis_L"]] = 150 * light * np.clip(left / 0.5, 0, 1)
        drive[gi["vis_R"]] = 150 * light * np.clip(right / 0.5, 0, 1)
        for s in self.arena.shadows:
            if s.target in self.ids:
                i = self.ids.index(s.target)
                p = min(1.0, (self.t - s.t_start) / s.duration)
                rel = np.angle(np.exp(1j * (s.direction - self.heading[i])))
                strength = 180 * p ** 2 * (0.3 + 0.7 * light)
                drive[gi["loom_L"], i] = strength * (1.0 if rel > -0.3 else 0.3)
                drive[gi["loom_R"], i] = strength * (1.0 if rel < 0.3 else 0.3)
        for f in self.arena.food:
            on = np.hypot(self.x - f.x, self.y - f.y) < f.radius
            drive[gi["sugar"], on] = 150 * (0.5 + np.clip(1 - self.energy[on], 0, 1)) * f.strength
        s = cfg.arena.size
        near = (self.x < 3) | (self.x > s - 3) | (self.y < 3) | (self.y > s - 3)
        wall_dir = np.arctan2(np.clip(self.y, 3, s - 3) - self.y, np.clip(self.x, 3, s - 3) - self.x) + np.pi
        rel = np.angle(np.exp(1j * (wall_dir - self.heading)))
        drive[gi["touch_L"], near & (rel >= 0)] = 100
        drive[gi["touch_R"], near & (rel < 0)] = 100
        for name, gene in self.group_gene.items():
            if gene:
                drive[gi[name]] *= self.genome[:, GENES.index(gene)]
        return torch.from_numpy(drive).to(self.dev)[self.sense_col]

    def _olfaction(self) -> torch.Tensor:
        B = len(self.ids)
        conc = np.zeros((B, len(self.olf.names), 2), dtype=np.float32)
        for s, sign in ((0, 1), (1, -1)):   # left, right antenna
            ax = self.x + 1.2 * np.cos(self.heading) - sign * 0.8 * np.sin(self.heading)
            ay = self.y + 1.2 * np.sin(self.heading) + sign * 0.8 * np.cos(self.heading)
            for o, name in enumerate(self.olf.names):
                if name == "fly":
                    d2 = (ax[:, None] - self.x[None]) ** 2 + (ay[:, None] - self.y[None]) ** 2
                    np.fill_diagonal(d2, np.inf)
                    conc[:, o, s] = np.exp(-d2 / (2 * 6.0 ** 2)).sum(1)
                else:
                    conc[:, o, s] = self.arena.odor(name, ax, ay)
        return self.olf.rates(torch.from_numpy(conc).to(self.dev), WORLD_DT * 1000)

    def _body(self, light: float):
        cfg, dt = self.cfg, WORLD_DT
        r = self.read
        B = len(self.ids)
        # Single DNa01/DNa02 neurons in this one connectome carry a static
        # left bias (symmetric odor drives left DNa02 much more). The body
        # adapts to a sustained left-right difference and steers on its changes.
        diff = (r["DNa01 L"] + r["DNa02 L"]) - (r["DNa01 R"] + r["DNa02 R"])
        self.turn_base += (diff - self.turn_base) * (dt / cfg.turn_adapt)
        turn = cfg.turn_gain * (diff - self.turn_base)
        on_food = np.zeros(B, dtype=bool)
        for f in self.arena.food:
            on_food |= np.hypot(self.x - f.x, self.y - f.y) < f.radius
        feeding = on_food & (r["MN9"] > cfg.feed_threshold)
        backward = r["MDN"] > cfg.mdn_threshold
        self.cooldown = np.maximum(self.cooldown - dt, 0)
        takeoff = (r["GF"] > cfg.gf_threshold) & (self.cooldown <= 0) & (self.airborne <= 0)

        walk = cfg.walk_speed * self.genome[:, GENES.index("walk")]
        speed = walk * (0.35 + 0.65 * light)
        speed[feeding] = 0
        speed[backward] = -0.5 * walk[backward]
        self.heading += turn * dt + cfg.wander * np.sqrt(dt) * self.rng.normal(0, 1, B)
        for i in np.flatnonzero(takeoff):
            self.airborne[i] = 0.15
            self.cooldown[i] = 1.0
            self.jumped_at[i] = self.t
            self.heading[i] += self.rng.uniform(-1.0, 1.0)
            self.energy[i] -= cfg.jump_cost
            self._event(i, "jump", "взлетела (гигантское волокно)")
        flying = self.airborne > 0
        speed[flying] = 130.0
        self.airborne = np.maximum(self.airborne - dt, 0)

        s = cfg.arena.size
        nx = self.x + speed * np.cos(self.heading) * dt
        ny = self.y + speed * np.sin(self.heading) * dt
        hit = (nx < 1) | (nx > s - 1) | (ny < 1) | (ny > s - 1)
        self.x, self.y = np.clip(nx, 1, s - 1), np.clip(ny, 1, s - 1)
        # bounce off walls towards the arena instead of sliding along them into corners
        if hit.any():
            to_center = np.arctan2(s / 2 - self.y[hit], s / 2 - self.x[hit])
            self.heading[hit] = to_center + self.rng.uniform(-0.8, 0.8, hit.sum())

        for f in self.arena.food:
            here = feeding & (np.hypot(self.x - f.x, self.y - f.y) < f.radius)
            if here.any() and f.sugar > 0:
                eat = min(f.sugar, cfg.sugar_per_s * dt * here.sum())
                f.sugar -= eat
                self.energy[here] += eat / here.sum() * cfg.energy_per_sugar
        for i in np.flatnonzero(feeding & (self.t - self.last_feed > 3.0)):
            self._event(i, "feed", "ест сахар (MN9 → хоботок)")
        self.last_feed[feeding] = self.t
        # a mature egg is laid on the food the fly chose to feed on
        can_lay = feeding & (self.age > cfg.maturity) & (self.t - self.last_egg > cfg.egg_interval) \
            & (self.energy > 0.5)
        for i in np.flatnonzero(can_lay):
            self.eggs.append(Egg(float(self.x[i]), float(self.y[i]), self.t, self.ids[i],
                                 int(self.generation[i]) + 1, self.genome[i].copy()))
            self.energy[i] -= cfg.egg_energy
            self.last_egg[i] = self.t
            self._event(i, "egg", "отложила яйцо")
        self.state = np.where(flying, 3, np.where(feeding, 1, np.where(backward, 2, 0)))

        self.energy -= cfg.metabolism * dt + cfg.move_cost * np.abs(speed) * dt * (~flying)
        self.energy = np.clip(self.energy, 0, 1)
        self.age += dt

        for sh in list(self.arena.shadows):
            if sh.target in self.ids and self.t - sh.t_start >= sh.duration and not getattr(sh, "resolved", False):
                i = self.ids.index(sh.target)
                sh.resolved = True
                if self.jumped_at[i] >= sh.t_start:
                    self._event(i, "escape", "увернулась от птицы")
                else:
                    self._kill(i, "bird", "поймана птицей")
        sx, sy, sr = cfg.arena.spider
        in_web = np.hypot(self.x - sx, self.y - sy) < sr
        for i in np.flatnonzero(in_web & (self.rng.random(B) < cfg.arena.spider_catch * dt)):
            self._kill(i, "spider", "поймана пауком")
        for i in np.flatnonzero(self.energy <= 0):
            self._kill(i, "starved", "умерла от голода")
        for i in np.flatnonzero(self.age >= self.lifespan):
            self._kill(i, "old", "умерла от старости")
        self._remove_dead()

    # --- bookkeeping --------------------------------------------------------
    def _event(self, i: int, kind: str, text: str):
        self.events.append({"t": round(self.t, 2), "fly": self.ids[i], "kind": kind, "text": text})

    def _kill(self, i: int, kind: str, text: str):
        if self.ids[i] not in {d for d, _ in self.dead}:
            self._event(i, "death_" + kind, text)
            self.dead.append((self.ids[i], kind))

    def _remove_dead(self):
        gone = {d for d, _ in self.dead}
        keep = np.array([fid not in gone for fid in self.ids])
        if keep.all():
            return
        cols = np.flatnonzero(keep)
        self.brain.keep(torch.tensor(cols, device=self.dev))
        self.olf.keep(torch.tensor(cols, device=self.dev))
        for c in self._columns:
            setattr(self, c, getattr(self, c)[cols])
        self.read = {k: v[cols] for k, v in self.read.items()}
        self.ids = [fid for fid, k in zip(self.ids, keep) if k]

    def frame(self) -> dict:
        return {
            **self.arena.snapshot(self.t),
            "eggs": [[round(e.x, 1), round(e.y, 1), round((self.t - e.laid) / self.cfg.hatch_time, 2)] for e in self.eggs],
            "flies": [[fid, round(float(self.x[i]), 1), round(float(self.y[i]), 1), round(float(self.heading[i]), 2),
                       int(self.state[i]), round(float(self.energy[i]), 3), round(float(self.age[i]), 1),
                       [int(self.read[k][i]) for k in READOUT], int(self.generation[i]), int(self.parent[i]),
                       [round(float(v), 2) for v in self.genome[i]]]
                      for i, fid in enumerate(self.ids)],
        }
