"""Running a batch of maze episodes with fly brains in the loop.

Every episode has a Nose brain (walks, smells) and, in team levels, an Eye
brain (sees along corridors). Each tick both brains get their senses as
Poisson input for `tick_ms`, then descending-neuron activity is read out as
an action (Nose) and a message symbol (both).
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from . import body, world
from .brain import FlyBrain
from .config import LIFParams

N_ACTIONS = 3  # world.FORWARD, LEFT, RIGHT


@dataclass
class Level:
    name: str
    maze: world.MazeConfig
    team: bool
    max_ticks: int
    slots: int = 1   # symbols per message


LEVELS = [
    Level("1 nose alone, tiny maze", world.MazeConfig(3, 3, loops=0.3, n_traps=0, min_food_dist=4), False, 30),
    # Smell goes through walls, so alone the Nose only knows direction, not the route;
    # from here on the Eye (who sees corridors) has a reason to talk.
    Level("2 team, small maze", world.MazeConfig(5, 5, loops=0.3, n_traps=0, min_food_dist=8), True, 50),
    # Traps: two-symbol messages, room for e.g. "direction + danger".
    Level("3 team, traps", world.MazeConfig(5, 5, loops=0.3, n_traps=2, min_food_dist=8), True, 50, slots=2),
    Level("4 team, big maze", world.MazeConfig(8, 8, loops=0.2, n_traps=4, min_food_dist=10), True, 80, slots=2),
]


def sym_param(role: str, slot: int, part: str) -> str:
    return f"{role}_sym{'' if slot == 0 else slot + 1}_{part}"


class Policy:
    """Layout of one parameter vector (per fly pair)."""

    def __init__(self, n_pools: int, slots: int = body.MAX_SLOTS):
        k, p = body.N_SYMBOLS, n_pools
        hear = body.N_SYMBOLS * slots
        self.shapes = {
            "nose_gain": (len(body.NOSE_FEATURES) - len(body.HEAR_FEATURES) + hear,),
            "eye_gain": (len(body.EYE_FEATURES) - len(body.HEAR_FEATURES) + hear,),
            "nose_act_w": (N_ACTIONS, p), "nose_act_b": (N_ACTIONS,),
        }
        for slot in range(slots):
            for role in ("nose", "eye"):
                self.shapes[sym_param(role, slot, "w")] = (k, p)
                self.shapes[sym_param(role, slot, "b")] = (k,)
        self.size = sum(int(np.prod(s)) for s in self.shapes.values())

    def slices(self) -> dict:
        out, i = {}, 0
        for name, shape in self.shapes.items():
            n = int(np.prod(shape))
            out[name] = slice(i, i + n)
            i += n
        return out

    def migrate(self, theta_old: np.ndarray, old_shapes: dict, into: np.ndarray) -> np.ndarray:
        """Carry parameters over by name from an older layout into `into` (new-layout vector)."""
        theta = into.copy()
        new, i = self.slices(), 0
        for name, shape in old_shapes.items():
            n = int(np.prod(shape))
            if name in new:
                if len(shape) == 1:  # gains: new features are appended at the end
                    theta[new[name].start:new[name].start + n] = theta_old[i:i + n]
                else:
                    theta[new[name]] = theta_old[i:i + n]
            i += n
        return theta

    def active_mask(self, level: "Level") -> np.ndarray:
        """Parameters that affect behaviour at this level; others are not perturbed."""
        mask = np.zeros(self.size, dtype=bool)
        s = self.slices()
        n_base_nose = len(body.NOSE_FEATURES) - len(body.HEAR_FEATURES)
        n_base_eye = len(body.EYE_FEATURES) - len(body.HEAR_FEATURES)
        hear = body.N_SYMBOLS * (level.slots if level.team else 0)
        mask[s["nose_gain"].start:s["nose_gain"].start + n_base_nose + hear] = True
        mask[s["nose_act_w"]] = mask[s["nose_act_b"]] = True
        if level.team:
            mask[s["eye_gain"].start:s["eye_gain"].start + n_base_eye + hear] = True
            for slot in range(level.slots):
                for role in ("nose", "eye"):
                    mask[s[sym_param(role, slot, "w")]] = mask[s[sym_param(role, slot, "b")]] = True
        return mask

    def unpack(self, theta: torch.Tensor) -> dict:
        """theta: (batch, size) -> dict of (batch, *shape) tensors."""
        out, i = {}, 0
        for name, shape in self.shapes.items():
            n = int(np.prod(shape))
            out[name] = theta[:, i:i + n].reshape(-1, *shape)
            i += n
        return out

    def init(self, rng) -> np.ndarray:
        theta = rng.normal(0, 0.1, self.size)
        i = 0
        for name, shape in self.shapes.items():
            n = int(np.prod(shape))
            if name.endswith("gain"):
                theta[i:i + n] = 0.0  # log-gain: gain 1
            i += n
        return theta


@dataclass
class Episode:
    map: world.Map
    pos: tuple
    heading: int
    dist_to_food: dict
    d0: int
    best: int
    outcome: str = "running"   # running | food | trap | timeout
    ticks: int = 0
    bumps: int = 0
    trace: list = field(default_factory=list)

    @property
    def done(self):
        return self.outcome != "running"

    def fitness(self, max_ticks: int) -> float:
        progress = 0.5 * (self.d0 - self.best) / max(self.d0, 1)
        if self.outcome == "food":
            return 1.0 + 0.5 * (1 - self.ticks / max_ticks)
        penalty = 0.5 if self.outcome == "trap" else 0.0
        return progress - penalty - 0.002 * self.bumps


class Runner:
    def __init__(self, con, wiring: body.Wiring, device=None, dt: float = 0.5, tick_ms: float = 50.0):
        self.con = con
        self.params = LIFParams(dt=dt)
        self.steps_per_tick = round(tick_ms / dt)
        self.tick_ms = tick_ms
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.io = body.IO(wiring, self.device)
        self.brain = None

    def _brain(self, batch: int) -> FlyBrain:
        if self.brain is None or self.brain.batch != batch:
            self.brain = FlyBrain(self.con, batch=batch, params=self.params, device=self.device)
        else:
            self.brain.reset()
        return self.brain

    @torch.no_grad()
    def run(self, theta: torch.Tensor, seeds: list[int], level: Level,
            mute_hearing: torch.Tensor | None = None) -> list[Episode]:
        """theta: (E, size), one row per episode. mute_hearing: (E,) bool."""
        E = len(seeds)
        policy = Policy(self.io.wiring.n_pools)
        P = policy.unpack(theta.to(self.device))
        eps = []
        for s in seeds:
            m = world.generate(s, level.maze)
            dist = world.bfs(m.grid, m.food)
            eps.append(Episode(m, m.start, m.heading, dist, dist[m.start], dist[m.start]))

        team, slots, k = level.team, level.slots, body.N_SYMBOLS
        brain = self._brain(2 * E if team else E)
        io = self.io
        n_readout = len(io.readout_idx)
        # (E, slots) symbols said on the previous tick
        sym_nose = torch.zeros(E, slots, dtype=torch.long, device=self.device)
        sym_eye = torch.zeros(E, slots, dtype=torch.long, device=self.device)
        hear_on = torch.ones(E, 1, device=self.device)
        if mute_hearing is not None:
            hear_on[mute_hearing.to(self.device)] = 0.0
        n_hear = len(body.HEAR_FEATURES)

        def heard(symbols):
            """(E, slots) partner symbols -> (E, n_hear) one-hot hearing features."""
            out = torch.zeros(E, n_hear, device=self.device)
            if team:
                for s in range(slots):
                    out[:, s * k:(s + 1) * k] = torch.nn.functional.one_hot(symbols[:, s], k) * hear_on
            return out

        for tick in range(level.max_ticks):
            if all(e.done for e in eps):
                break
            nose_f = torch.tensor([self._nose_features(e) for e in eps], dtype=torch.float32, device=self.device)
            nose_f[:, -n_hear:] = heard(sym_eye)
            drive = [nose_f * P["nose_gain"].exp()]
            if team:
                eye_f = torch.tensor([self._eye_features(e) for e in eps], dtype=torch.float32, device=self.device)
                eye_f[:, -n_hear:] = heard(sym_nose)
                drive.append(eye_f * P["eye_gain"].exp())
            rates = torch.cat([io.rates("nose", drive[0])] + ([io.rates("eye", drive[1])] if team else []), dim=1)

            counts = torch.zeros(n_readout, brain.batch, device=self.device)
            for _ in range(self.steps_per_tick):
                counts += brain.step(io.input_idx, rates)[io.readout_idx]
            z = io.pooled(counts) * (1000.0 / self.tick_ms) / 20.0  # ~Hz / 20
            z_nose, z_eye = z[:E], z[E:]

            act = (P["nose_act_w"] @ z_nose.unsqueeze(2)).squeeze(2) + P["nose_act_b"]
            actions = act.argmax(1).tolist()
            if team:
                say = lambda role, z_, s: ((P[sym_param(role, s, "w")] @ z_.unsqueeze(2)).squeeze(2)
                                           + P[sym_param(role, s, "b")]).argmax(1)
                sym_nose = torch.stack([say("nose", z_nose, s) for s in range(slots)], 1)
                sym_eye = torch.stack([say("eye", z_eye, s) for s in range(slots)], 1)
            sn, se = sym_nose.tolist(), sym_eye.tolist()

            for i, e in enumerate(eps):
                if e.done:
                    continue
                pos, heading, event = world.step(e.map, e.pos, e.heading, actions[i])
                # symbols: tuple per slot, None when flies don't talk
                e.trace.append((e.pos, e.heading, actions[i], tuple(sn[i]) if team else None,
                                tuple(se[i]) if team else None, event))
                e.pos, e.heading = pos, heading
                e.ticks = tick + 1
                if event == "bump":
                    e.bumps += 1
                e.best = min(e.best, e.dist_to_food.get(pos, e.best))
                if event in ("food", "trap"):
                    e.outcome = event
        for e in eps:
            if not e.done:
                e.outcome = "timeout"
        return eps

    @staticmethod
    def _nose_features(e: Episode) -> list[float]:
        left, right = world.smell(e.map, e.pos, e.heading)
        diff = (left - right) * 20.0
        bumped = 1.0 if e.trace and e.trace[-1][5] == "bump" else 0.0
        return [left, right, max(diff, 0.0), max(-diff, 0.0), bumped] + [0.0] * len(body.HEAR_FEATURES)

    @staticmethod
    def _eye_features(e: Episode) -> list[float]:
        rng = e.map.cfg.vision_range
        out = []
        for ray in world.vision(e.map, e.pos, e.heading):
            out.append(ray.wall_dist / rng)
            out.append(0.0 if ray.trap_dist is None else 1.0 - (ray.trap_dist - 1) / rng)
            out.append(0.0 if ray.food_dist is None else 1.0 - (ray.food_dist - 1) / rng)
        return out + [0.0] * len(body.HEAR_FEATURES)
