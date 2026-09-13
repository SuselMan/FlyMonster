"""Flies with whole-brain simulations living in the arena.

Loop every world step (10 ms): world -> sensory rates -> 20 brain steps
(dt 0.5 ms) -> descending-neuron spike counts -> low-pass readout -> body.

Boundaries (see docs in flysim/life/README section of the main README):
- Sensory transduction and the body are our models (category C). Everything
  between sensory neurons and descending/motor neurons is the connectome.
- Body mapping uses only descending neurons with published roles:
  DNa01/DNa02 ipsilateral turning, MDN backward walking, DNp01 giant fiber
  takeoff, MN9 proboscis (feeding). Walking itself is an innate generator
  (the leg circuits live in the ventral nerve cord, absent from FAFB).
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


@dataclass
class LifeConfig:
    n_flies: int = 24
    seed: int = 1
    walk_speed: float = 14.0      # mm/s, innate walking generator
    turn_gain: float = 0.025      # rad/s per Hz of left-right DNa01+DNa02 difference
    turn_adapt: float = 5.0       # s, adaptation of the body to a sustained left-right difference
    wander: float = 1.2           # rad/s^0.5, turning noise of the walking generator
    gf_threshold: float = 60.0    # Hz, low-passed giant fiber rate that triggers takeoff
    mdn_threshold: float = 25.0   # Hz, backward walking
    feed_threshold: float = 15.0  # Hz, MN9 while touching food
    metabolism: float = 0.004     # energy per s at rest (energy 1 = full)
    move_cost: float = 0.0004     # energy per mm walked
    jump_cost: float = 0.02
    sugar_per_s: float = 6.0      # food units eaten per s
    energy_per_sugar: float = 0.02
    lifespan: tuple = (2700.0, 3600.0)   # s, 45-60 min of simulated adult life
    physiology: Physiology = DEFAULT
    arena: ArenaConfig = field(default_factory=ArenaConfig)


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
        # body state, one row per living fly (kept in brain batch order)
        s = cfg.arena.size
        self.ids = list(range(B))
        self.next_id = B
        self.x = self.rng.uniform(20, s - 20, B)
        self.y = self.rng.uniform(20, s - 20, B)
        self.heading = self.rng.uniform(-np.pi, np.pi, B)
        self.energy = np.full(B, 0.7)
        self.age = np.zeros(B)
        self.lifespan = self.rng.uniform(*cfg.lifespan, B)
        self.airborne = np.zeros(B)
        self.cooldown = np.zeros(B)
        self.jumped_at = np.full(B, -1e9)
        self.state = np.zeros(B, dtype=int)   # 0 walk, 1 feed, 2 back, 3 jump
        self.turn_base = np.zeros(B)
        self.read = {k: np.zeros(B) for k in READOUT}
        self.events: list = []
        self.dead: list = []

    # --- wiring ---------------------------------------------------------------
    def _wire(self):
        m, con = self.meta, self.con
        ct = m.cell_type.fillna("").to_numpy()
        side = m.side.fillna("").to_numpy()
        sup = m.super_class.fillna("").to_numpy()
        sub = m.cell_sub_class.fillna("").to_numpy()
        dn = np.flatnonzero(sup == "descending")
        reach = dn_reach(con, dn)

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
        idx, col = [], []
        for g, name in enumerate(self.group_names):
            idx.extend(groups[name].tolist())
            col.extend([g] * len(groups[name]))
        self.sense_idx = torch.tensor(idx, device=self.dev)
        self.sense_col = torch.tensor(col, device=self.dev)
        self.readout_idx = {}
        for name, (t, s) in READOUT.items():
            if name == "MN9":
                sel = [con.index_of[neurons.MN9]]
            else:
                sel = np.flatnonzero((ct == t) & ((side == s) if s else True)).tolist()
            self.readout_idx[name] = torch.tensor(sel, device=self.dev)
        # hunger state acts on the NPF circuit's excitability (not as a sensory drive)
        self.npf_idx = torch.tensor(np.flatnonzero(np.char.find(ct.astype(str), "NPF") >= 0), device=self.dev)

    # --- one world step -----------------------------------------------------
    @torch.no_grad()
    def step(self):
        cfg, B = self.cfg, len(self.ids)
        if B == 0:
            return
        light = self.arena.light(self.t)
        rates = self._sensory_rates(light)                     # (n_sense, B)
        olf_rates = self._olfaction()                          # (n_orn, B)
        idx = torch.cat([self.sense_idx, self.olf.input_idx])
        all_rates = torch.cat([rates, olf_rates])
        hunger = torch.tensor(np.clip(1 - self.energy, 0, 1), dtype=torch.float32, device=self.dev)
        counts = {k: torch.zeros(B, device=self.dev) for k in READOUT}
        tau_g = self.brain.p.tau
        for _ in range(self.steps_per_world):
            if len(self.npf_idx):
                # depolarising offset ~ lowered threshold, scaled by hunger (mV)
                self.brain.g[self.npf_idx] += (4.0 * hunger) * (self.brain.p.dt / tau_g)
            spikes = self.brain.step(idx, all_rates)
            for k, sel in self.readout_idx.items():
                counts[k] += spikes[sel].float().mean(0) if len(sel) else 0
        a = WORLD_DT / READ_TAU
        for k in READOUT:
            hz = counts[k].cpu().numpy() / WORLD_DT
            self.read[k] += (hz - self.read[k]) * a
        self._body(light)
        self.arena.update(self.t, WORLD_DT, None, self.ids)
        self.t += WORLD_DT

    def _sensory_rates(self, light: float) -> torch.Tensor:
        cfg, B = self.cfg, len(self.ids)
        drive = np.zeros((len(self.group_names), B), dtype=np.float32)
        g = {n: i for i, n in enumerate(self.group_names)}
        # vision: salience of objects in each hemifield (other flies, food spots)
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
        drive[g["vis_L"]] = 150 * light * np.clip(left / 0.5, 0, 1)
        drive[g["vis_R"]] = 150 * light * np.clip(right / 0.5, 0, 1)
        # looming bird shadow
        for s in self.arena.shadows:
            if s.target in self.ids:
                i = self.ids.index(s.target)
                p = min(1.0, (self.t - s.t_start) / s.duration)
                rel = np.angle(np.exp(1j * (s.direction - self.heading[i])))
                strength = 180 * p ** 2 * (0.3 + 0.7 * light)
                drive[g["loom_L"], i] = strength * (1.0 if rel > -0.3 else 0.3)
                drive[g["loom_R"], i] = strength * (1.0 if rel < 0.3 else 0.3)
        # taste on contact with food, stronger when hungry
        for f in self.arena.food:
            on = np.hypot(self.x - f.x, self.y - f.y) < f.radius
            drive[g["sugar"], on] = 150 * (0.5 + np.clip(1 - self.energy[on], 0, 1)) * f.strength
        # touch: head bristles when pressing against a wall
        s = cfg.arena.size
        near = (self.x < 3) | (self.x > s - 3) | (self.y < 3) | (self.y > s - 3)
        wall_dir = np.arctan2(np.clip(self.y, 3, s - 3) - self.y, np.clip(self.x, 3, s - 3) - self.x) + np.pi
        rel = np.angle(np.exp(1j * (wall_dir - self.heading)))
        drive[g["touch_L"], near & (rel >= 0)] = 100
        drive[g["touch_R"], near & (rel < 0)] = 100
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
        # left bias (e.g. symmetric odor drives left DNa02 much more). The body
        # model adapts to sustained left-right difference (time constant
        # turn_adapt) and steers on its changes. Our assumption, category C.
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

        speed = np.full(B, cfg.walk_speed) * (0.35 + 0.65 * light)   # slower in the dark
        speed[feeding] = 0
        speed[backward] = -0.5 * cfg.walk_speed
        self.heading += turn * dt + cfg.wander * np.sqrt(dt) * self.rng.normal(0, 1, B)
        # takeoff: a short ballistic flight away
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
        self.heading[hit] += np.pi / 2 * self.rng.choice([-1, 1], hit.sum()) * 0.3

        # feeding
        for f in self.arena.food:
            here = feeding & (np.hypot(self.x - f.x, self.y - f.y) < f.radius)
            if here.any() and f.sugar > 0:
                eat = min(f.sugar, cfg.sugar_per_s * dt * here.sum())
                f.sugar -= eat
                self.energy[here] += eat / here.sum() * cfg.energy_per_sugar
        starting = feeding & (self.state != 1)
        for i in np.flatnonzero(starting):
            self._event(i, "feed", "ест сахар (MN9 → хоботок)")
        self.state = np.where(flying, 3, np.where(feeding, 1, np.where(backward, 2, 0)))

        # metabolism and ageing
        self.energy -= cfg.metabolism * dt + cfg.move_cost * np.abs(speed) * dt * (~flying)
        self.energy = np.clip(self.energy, 0, 1)
        self.age += dt

        # dangers
        for sh in list(self.arena.shadows):
            if sh.target in self.ids and self.t - sh.t_start >= sh.duration and not getattr(sh, "resolved", False):
                i = self.ids.index(sh.target)
                sh.resolved = True
                if self.jumped_at[i] >= sh.t_start:
                    self._event(i, "escape", "увернулась от птицы")
                else:
                    self._kill(i, "bird", "поймана птицей")
        sx, sy, sr = self.cfg.arena.spider
        in_web = np.hypot(self.x - sx, self.y - sy) < sr
        caught = in_web & (self.rng.random(len(self.ids)) < 0.25 * dt)
        for i in np.flatnonzero(caught):
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
        if self.ids[i] not in [d for d, _ in self.dead]:
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
        for name in ("x", "y", "heading", "energy", "age", "lifespan", "airborne", "cooldown", "jumped_at", "state",
                     "turn_base"):
            setattr(self, name, getattr(self, name)[cols])
        self.read = {k: v[cols] for k, v in self.read.items()}
        self.ids = [fid for fid, k in zip(self.ids, keep) if k]

    def frame(self) -> dict:
        return {
            **self.arena.snapshot(self.t),
            "flies": [[fid, round(float(self.x[i]), 1), round(float(self.y[i]), 1), round(float(self.heading[i]), 2),
                       int(self.state[i]), round(float(self.energy[i]), 3), round(float(self.age[i]), 1),
                       [int(self.read[k][i]) for k in READOUT]]
                      for i, fid in enumerate(self.ids)],
        }
