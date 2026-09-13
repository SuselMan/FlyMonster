"""The whole fly brain playing Flappy Fly, with no trained readout.

Eyes: "gap above" drives left visual projection neurons, "gap below" the
right ones, an approaching pipe drives looming detectors (LC4, LPLC2).
Decision: read straight from the fly's own descending neurons. With
mode "dnb05" the fly flaps when left DNb05 fires more than right DNb05
(left visual input activates left DNb05 in this model).
Evolution may only tune input gains and the decision threshold.
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from . import body, connectome
from .brain import FlyBrain
from .config import LIFParams
from .flappy import FlappyConfig, Games

SENSES = ("gap_above", "gap_below", "pipe_near")
# Readout pairs (a, b): flap when spikes(a) - spikes(b) > threshold.
MODES = {
    "dnb05": (("DNb05", "left"), ("DNb05", "right")),
    "dna01": (("DNa01", "right"), ("DNa01", "left")),   # contralateral in the probe
    "giant_fiber": (("DNp01", "left"), None),
}


@dataclass
class FlyParams:
    gains: dict = field(default_factory=lambda: {"gap_above": 1.0, "gap_below": 1.0, "pipe_near": 0.0})
    threshold: float = 0.0
    max_hz: float = 100.0

    def to_vector(self) -> np.ndarray:
        return np.array([np.log(max(self.gains[s], 1e-3)) for s in SENSES] + [self.threshold])

    @staticmethod
    def from_vector(v) -> "FlyParams":
        return FlyParams({s: float(np.exp(v[i])) for i, s in enumerate(SENSES)}, float(v[len(SENSES)]))


class FlappyFly:
    def __init__(self, con=None, mode: str = "dnb05", group_size: int = 60, device=None,
                 dt: float = 0.5, tick_ms: float = 50.0):
        self.con = con or connectome.load()
        self.mode = mode
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.params = LIFParams(dt=dt)
        self.steps_per_tick = round(tick_ms / dt)
        ann = connectome.annotations()
        ann = ann[~ann.index.duplicated() & ann.index.isin(list(self.con.index_of))]
        dn = np.array(self.con.indices(ann.index[ann.super_class == "descending"]))
        reach = body.dn_reach(self.con, dn)

        def ranked(mask, k):
            idx = np.array(self.con.indices(ann.index[mask]))
            return [int(i) for i in idx[np.argsort(-reach[idx], kind="stable")][:k]]

        vpn = ann.super_class == "visual_projection"
        self.groups = {
            "gap_above": ranked(vpn & (ann.side == "left"), group_size),
            "gap_below": ranked(vpn & (ann.side == "right"), group_size),
            "pipe_near": [int(i) for i in self.con.indices(ann.index[ann.cell_type.isin(["LC4", "LPLC2"])])],
        }
        self.readout = {}
        for t, side in {x for pair in MODES.values() for x in pair if x}:
            self.readout[(t, side)] = [int(i) for i in self.con.indices(
                ann.index[(ann.super_class == "descending") & (ann.cell_type == t) & (ann.side == side)])]

        neurons = sorted({i for g in self.groups.values() for i in g})
        row = {n: r for r, n in enumerate(neurons)}
        self.input_idx = torch.tensor(neurons, device=self.device)
        self.input_map = torch.zeros(len(neurons), len(SENSES), device=self.device)
        for f, s in enumerate(SENSES):
            self.input_map[[row[i] for i in self.groups[s]], f] = 1.0
        self.read_idx = {k: torch.tensor(v, device=self.device) for k, v in self.readout.items()}
        self.brain = None

    @torch.no_grad()
    def play(self, seeds: list[int], params: list[FlyParams], cfg: FlappyConfig | None = None,
             record: bool = False):
        """One game per (seed, params) pair. Returns Games and, if record, per-game traces."""
        B = len(seeds)
        games = Games(seeds, cfg)
        if self.brain is None:
            self.brain = FlyBrain(self.con, batch=B, params=self.params, device=self.device)
        else:
            self.brain.resize(B)
        brain = self.brain
        gains = torch.tensor([[p.gains[s] for s in SENSES] for p in params], dtype=torch.float32,
                             device=self.device)
        thresholds = np.array([p.threshold for p in params])
        max_hz = torch.tensor([p.max_hz for p in params], dtype=torch.float32, device=self.device)
        a_key, b_key = MODES[self.mode]
        traces = [[] for _ in range(B)] if record else None
        live = np.arange(B)

        while games.alive.any():
            if len(live) - games.alive.sum() >= max(1, len(live) // 10):
                keep = games.alive[live]
                brain.keep(torch.tensor(np.flatnonzero(keep), device=self.device))
                live = live[keep]
            s = games.senses()
            feat = torch.tensor(np.stack([s[k][live] for k in SENSES], 1), dtype=torch.float32,
                                device=self.device)
            drive = (feat * gains[live]).clamp(0, 1) * max_hz[live, None]
            rates = self.input_map @ drive.T

            count = {k: torch.zeros(len(live), device=self.device) for k in self.read_idx}
            for _ in range(self.steps_per_tick):
                spikes = brain.step(self.input_idx, rates)
                for k, idx in self.read_idx.items():
                    count[k] += spikes[idx].sum(0)
            count = {k: v.cpu().numpy() for k, v in count.items()}
            signal = count[a_key] - (count[b_key] if b_key else 0)
            flap = np.zeros(B, dtype=bool)
            flap[live] = signal > thresholds[live]

            if record:
                for j, b in enumerate(live):
                    if games.alive[b]:
                        traces[b].append({
                            "y": round(float(games.y[b]), 4), "flap": bool(flap[b]),
                            **{k: round(float(s[k][b]), 3) for k in SENSES},
                            "dn": {f"{t} {side[0].upper()}": int(count[(t, side)][j]) for t, side in self.read_idx},
                        })
            games.step(flap)
        return games, traces
