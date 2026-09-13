"""Flappy Fly: a batch of seeded Flappy Bird games, one tick = one brain readout.

Coordinates: the screen is 1 high, y = 0 at the bottom. The fly stays at
x = FLY_X, pipes move left. A pipe is a column of width PIPE_W with a gap.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class FlappyConfig:
    gravity: float = 0.010      # per tick^2
    flap_v: float = 0.040       # vertical speed right after a flap
    max_fall: float = -0.06
    speed: float = 0.020        # pipe speed per tick
    pipe_w: float = 0.10
    spacing: float = 0.60       # between pipes
    gap: float = 0.34
    gap_margin: float = 0.22    # gap center stays in [margin, 1 - margin]
    first_pipe: float = 0.9
    fly_x: float = 0.25
    fly_r: float = 0.025        # collision radius
    max_ticks: int = 300


class Games:
    """B independent games advanced together."""

    def __init__(self, seeds: list[int], cfg: FlappyConfig | None = None, n_pipes: int = 64):
        self.cfg = c = cfg or FlappyConfig()
        self.B = len(seeds)
        self.seeds = list(seeds)
        # pipes[b, i] = (x at tick 0, gap center)
        self.pipe_x0 = c.first_pipe + c.spacing * np.arange(n_pipes)[None].repeat(self.B, 0)
        self.gap_y = np.stack([np.random.default_rng(s).uniform(c.gap_margin, 1 - c.gap_margin, n_pipes)
                               for s in seeds])
        self.y = np.full(self.B, 0.5)
        self.vy = np.zeros(self.B)
        self.alive = np.ones(self.B, dtype=bool)
        self.score = np.zeros(self.B, dtype=int)
        self.ticks = np.zeros(self.B, dtype=int)
        self.t = 0

    def pipe_x(self) -> np.ndarray:
        return self.pipe_x0 - self.cfg.speed * self.t

    def next_pipe(self):
        """Index, x distance (fly to pipe's near edge) and gap center of the first pipe not yet passed."""
        c = self.cfg
        x = self.pipe_x()
        ahead = x + c.pipe_w > c.fly_x - c.fly_r
        idx = ahead.argmax(1)
        rows = np.arange(self.B)
        return idx, x[rows, idx] - c.fly_x, self.gap_y[rows, idx]

    def senses(self) -> dict:
        """Signals in [0, 1] for the fly's eyes."""
        _, dist, gap = self.next_pipe()
        dy = gap - self.y
        return {
            "gap_above": np.clip(dy / 0.25, 0, 1),
            "gap_below": np.clip(-dy / 0.25, 0, 1),
            "pipe_near": np.clip(1 - np.maximum(dist, 0) / 0.6, 0, 1),
        }

    def step(self, flap: np.ndarray):
        c = self.cfg
        a = self.alive
        self.vy[a] = np.where(flap[a], c.flap_v, np.maximum(self.vy[a] - c.gravity, c.max_fall))
        self.y[a] += self.vy[a]
        self.t += 1
        self.ticks[a] = self.t

        x = self.pipe_x()
        inside = (x - c.fly_r < c.fly_x) & (c.fly_x < x + c.pipe_w + c.fly_r)
        outside_gap = np.abs(self.y[:, None] - self.gap_y) > c.gap / 2 - c.fly_r
        hit = (inside & outside_gap).any(1) | (self.y < c.fly_r) | (self.y > 1 - c.fly_r)
        passed = (x + c.pipe_w < c.fly_x - c.fly_r).sum(1)
        self.score[a] = passed[a]
        self.alive &= ~hit
        self.alive &= self.t < c.max_ticks

    def fitness(self) -> np.ndarray:
        return self.score + self.ticks / 100.0

    def pipes_json(self, b: int, n: int | None = None) -> list:
        n = n or int(self.score[b]) + 3
        return [[float(self.pipe_x0[b, i]), float(self.gap_y[b, i])] for i in range(n)]
