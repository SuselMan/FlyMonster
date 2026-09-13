"""2D arena seen from above: walls, food patches with odor, a spider corner, bird shadows.

Units: millimetres and seconds. A fly is ~3 mm long and walks ~10-20 mm/s.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Food:
    x: float
    y: float
    sugar: float            # remaining amount (arbitrary units)
    radius: float = 4.0
    odor: str = "fruit"

    @property
    def strength(self) -> float:
        return min(1.0, self.sugar / 50.0)


@dataclass
class Shadow:
    """A bird diving at a fly: angular size grows until it arrives."""
    target: int             # fly id
    t_start: float
    duration: float = 0.8   # s from appearance to strike
    direction: float = 0.0  # world angle it comes from (rad)


@dataclass
class ArenaConfig:
    size: float = 160.0
    n_food: int = 4
    food_sugar: float = 400.0
    odor_sigma: float = 30.0          # mm, width of the odor plume
    spider: tuple = (140.0, 140.0, 18.0)   # x, y, radius of the spider's corner
    shadow_rate: float = 1 / 40.0     # attacks per fly per second (in daylight)
    day_length: float = 600.0         # s of simulated time for a full day/night cycle
    food_regrow: float = 90.0         # s until a depleted patch reappears elsewhere


@dataclass
class Arena:
    cfg: ArenaConfig
    rng: np.random.Generator
    food: list = field(default_factory=list)
    shadows: list = field(default_factory=list)
    regrow_at: list = field(default_factory=list)

    @staticmethod
    def create(cfg: ArenaConfig, seed: int) -> "Arena":
        a = Arena(cfg, np.random.default_rng(seed))
        for _ in range(cfg.n_food):
            a.food.append(a._new_food())
        return a

    def _new_food(self) -> Food:
        m = 25.0
        while True:
            x, y = self.rng.uniform(m, self.cfg.size - m, 2)
            sx, sy, sr = self.cfg.spider
            if np.hypot(x - sx, y - sy) > sr + 20:
                return Food(float(x), float(y), self.cfg.food_sugar)

    def light(self, t: float) -> float:
        """1 at noon, ~0.1 at midnight."""
        phase = (t % self.cfg.day_length) / self.cfg.day_length
        return float(0.55 + 0.45 * np.cos(2 * np.pi * (phase - 0.25)))

    def odor(self, name: str, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Concentration of an odorant at points (vectorised)."""
        c = np.zeros_like(px)
        s2 = 2 * self.cfg.odor_sigma ** 2
        if name == "spider":
            sx, sy, sr = self.cfg.spider
            return 1.0 * np.exp(-((px - sx) ** 2 + (py - sy) ** 2) / s2)
        for f in self.food:
            if f.odor == name and f.sugar > 0:
                c += f.strength * np.exp(-((px - f.x) ** 2 + (py - f.y) ** 2) / s2)
        return c

    def update(self, t: float, dt: float, n_flies_alive: np.ndarray, fly_ids: list):
        # depleted patches vanish and regrow somewhere else later
        for f in list(self.food):
            if f.sugar <= 0:
                self.food.remove(f)
                self.regrow_at.append(t + self.cfg.food_regrow)
        for when in list(self.regrow_at):
            if t >= when:
                self.regrow_at.remove(when)
                self.food.append(self._new_food())
        # birds attack more in daylight
        self.shadows = [s for s in self.shadows if t - s.t_start <= s.duration + 0.2]
        rate = self.cfg.shadow_rate * self.light(t)
        for fid in fly_ids:
            if self.rng.random() < rate * dt and not any(s.target == fid for s in self.shadows):
                self.shadows.append(Shadow(fid, t, direction=float(self.rng.uniform(-np.pi, np.pi))))

    def snapshot(self, t: float) -> dict:
        return {
            "t": round(t, 3), "light": round(self.light(t), 3),
            "food": [[round(f.x, 1), round(f.y, 1), round(f.sugar, 1)] for f in self.food],
            "shadows": [[s.target, round((t - s.t_start) / s.duration, 3), round(s.direction, 2)] for s in self.shadows],
        }
