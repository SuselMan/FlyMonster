"""2D forest-floor world seen from above.

Units: millimetres and seconds. A fly is ~3 mm long and walks ~10-20 mm/s.
Everything here is the world model (scripted); only the flies have brains.

- Fruits fall from the canopy, ripen into rot over minutes (odor shifts from
  "fruit" to "vinegar" as they ferment) and vanish when rotten or eaten.
- Wind slowly changes; odor plumes are stretched downwind.
- Water and stones block walking but not flying.
- A spider waits in a sticky web and walks to stuck flies.
- A centipede roams and hunts flies it sees (cone) or feels (vibration).
- Birds dive at flies in daylight.
- Viewers can drop fruit through a commands file.
"""
import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Fruit:
    id: int
    x: float
    y: float
    sugar: float
    born: float
    rot_time: float = 480.0     # s until fully fermented
    radius: float = 6.0
    by_viewer: bool = False

    def freshness(self, t: float) -> float:
        return float(np.clip(1 - (t - self.born) / self.rot_time, 0, 1))

    def strength(self) -> float:
        return float(min(1.0, self.sugar / 60.0))


@dataclass
class Obstacle:
    kind: str          # "water" | "stone"
    x: float
    y: float
    rx: float
    ry: float
    angle: float = 0.0

    def inside(self, px, py, pad: float = 0.0):
        c, s = np.cos(-self.angle), np.sin(-self.angle)
        dx, dy = px - self.x, py - self.y
        u, v = dx * c - dy * s, dx * s + dy * c
        return (u / (self.rx + pad)) ** 2 + (v / (self.ry + pad)) ** 2 < 1


@dataclass
class Shadow:
    """A bird diving at a fly: angular size grows until it arrives."""
    target: int
    t_start: float
    duration: float = 0.8
    direction: float = 0.0
    resolved: bool = False


@dataclass
class Spider:
    home_x: float
    home_y: float
    web_r: float
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    target: int | None = None
    speed: float = 6.0


@dataclass
class Centipede:
    x: float
    y: float
    heading: float
    target: int | None = None
    body: list = field(default_factory=list)   # trailing points for drawing
    wander_speed: float = 10.0
    hunt_speed: float = 24.0
    see_range: float = 45.0
    see_half_angle: float = 1.0
    feel_range: float = 12.0


@dataclass
class ArenaConfig:
    width: float = 600.0
    height: float = 400.0
    fruit_target: int = 9            # fruits the canopy tends to keep on the ground
    fruit_rate: float = 1 / 45.0     # fruits falling per second (if below target)
    fruit_sugar: float = 500.0
    odor_sigma: float = 30.0
    shadow_rate: float = 1 / 25.0    # bird attacks per second on the whole world (daylight)
    day_length: float = 600.0
    seed_obstacles: int = 11


@dataclass
class Arena:
    cfg: ArenaConfig
    rng: np.random.Generator
    fruits: list = field(default_factory=list)
    obstacles: list = field(default_factory=list)
    shadows: list = field(default_factory=list)
    spider: Spider | None = None
    centipede: Centipede | None = None
    wind: np.ndarray = field(default_factory=lambda: np.zeros(2))
    next_fruit_id: int = 0
    log: list = field(default_factory=list)     # world events for the feed

    @staticmethod
    def create(cfg: ArenaConfig, seed: int) -> "Arena":
        a = Arena(cfg, np.random.default_rng(seed))
        W, H = cfg.width, cfg.height
        # proportions follow the viewer sprites (lake ~1.15:1, stones roughly round)
        a.obstacles = [
            Obstacle("water", W * 0.52, H * 0.55, 72, 60, 0.0),
            Obstacle("water", W * 0.17, H * 0.24, 40, 34, 0.0),
            Obstacle("stone", W * 0.35, H * 0.74, 14, 14, 0.0),
            Obstacle("stone", W * 0.80, H * 0.27, 19, 19, 0.0),
            Obstacle("stone", W * 0.84, H * 0.80, 12, 12, 0.0),
            Obstacle("stone", W * 0.10, H * 0.72, 16, 16, 0.0),
            Obstacle("stone", W * 0.64, H * 0.14, 11, 11, 0.0),
        ]
        a.spider = Spider(W * 0.30, H * 0.42, 22.0)
        a.spider.x, a.spider.y = a.spider.home_x, a.spider.home_y
        a.centipede = Centipede(W * 0.85, H * 0.55, float(a.rng.uniform(-np.pi, np.pi)))
        for _ in range(6):
            a.drop_fruit(0.0, announce=False, age=float(a.rng.uniform(0, 200)))
        return a

    # --- geometry --------------------------------------------------------------
    def blocked(self, px, py, pad: float = 0.0):
        out = np.zeros_like(np.asarray(px, dtype=float), dtype=bool)
        for o in self.obstacles:
            out |= o.inside(px, py, pad)
        return out

    def free_spot(self, margin: float = 20.0):
        for _ in range(200):
            x = float(self.rng.uniform(margin, self.cfg.width - margin))
            y = float(self.rng.uniform(margin, self.cfg.height - margin))
            if not self.blocked(x, y, pad=8) and np.hypot(x - self.spider.home_x, y - self.spider.home_y) > self.spider.web_r + 15:
                return x, y
        return self.cfg.width / 2, self.cfg.height / 2

    def in_web(self, px, py):
        s = self.spider
        return np.hypot(px - s.home_x, py - s.home_y) < s.web_r

    # --- fruits ---------------------------------------------------------------
    def drop_fruit(self, t: float, x=None, y=None, sugar=None, announce=True, by_viewer=False, age=0.0):
        if x is None:
            x, y = self.free_spot()
        f = Fruit(self.next_fruit_id, float(x), float(y), float(sugar or self.cfg.fruit_sugar), t - age,
                  radius=6.0 * float(np.sqrt((sugar or self.cfg.fruit_sugar) / self.cfg.fruit_sugar)), by_viewer=by_viewer)
        self.next_fruit_id += 1
        self.fruits.append(f)
        if announce:
            self.log.append({"t": round(t, 2), "kind": "fruit_drop", "x": round(f.x), "y": round(f.y),
                             "text": "зритель положил плод" if by_viewer else "с дерева упал плод"})
        return f

    def light(self, t: float) -> float:
        phase = (t % self.cfg.day_length) / self.cfg.day_length
        return float(0.55 + 0.45 * np.cos(2 * np.pi * (phase - 0.25)))

    def odor(self, name: str, px: np.ndarray, py: np.ndarray, t: float) -> np.ndarray:
        """Concentration at points. Plumes are stretched downwind."""
        c = np.zeros_like(px, dtype=float)
        wx, wy = self.wind
        ws = float(np.hypot(wx, wy))
        ux, uy = (wx / ws, wy / ws) if ws > 1e-6 else (1.0, 0.0)
        sig = self.cfg.odor_sigma

        def plume(sx, sy, amount, sigma):
            if np.ndim(sx):
                dx, dy = px[..., None] - sx, py[..., None] - sy
            else:
                dx, dy = px - sx, py - sy
            along, cross = dx * ux + dy * uy, -dx * uy + dy * ux
            s_along = np.where(along > 0, sigma * (1 + 2.5 * ws), sigma)   # longer downwind
            return amount * np.exp(-(along ** 2) / (2 * s_along ** 2) - cross ** 2 / (2 * sigma ** 2))

        if name in ("fruit", "vinegar"):
            if self.fruits:   # all fruits at once: sources along a trailing axis
                fx = np.array([f.x for f in self.fruits])
                fy = np.array([f.y for f in self.fruits])
                fr = np.array([f.freshness(t) for f in self.fruits])
                amount = np.array([f.strength() for f in self.fruits]) * (fr if name == "fruit" else (1 - fr) * 1.2)
                c = plume(fx, fy, amount, sig).sum(-1)
        elif name == "spider":
            c += plume(self.spider.home_x, self.spider.home_y, 0.8, sig * 0.7)
        elif name == "centipede":
            c += plume(self.centipede.x, self.centipede.y, 1.0, sig * 0.6)
        return c

    # --- world update -----------------------------------------------------------
    def update(self, t: float, dt: float, flies: dict):
        """flies: {"ids", "x", "y", "stuck", "airborne", "moving"} arrays of living flies."""
        cfg = self.cfg
        # wind drifts slowly
        self.wind += self.rng.normal(0, 0.03, 2) * np.sqrt(dt) - self.wind * 0.02 * dt
        self.wind = np.clip(self.wind, -0.8, 0.8)
        # fruits fall, rot, disappear
        if len(self.fruits) < cfg.fruit_target and self.rng.random() < cfg.fruit_rate * dt:
            self.drop_fruit(t)
        for f in list(self.fruits):
            if f.sugar <= 1 or t - f.born > f.rot_time * 1.6:
                self.fruits.remove(f)
                self.log.append({"t": round(t, 2), "kind": "fruit_gone", "x": round(f.x), "y": round(f.y),
                                 "text": "плод съеден" if f.sugar <= 1 else "плод сгнил и исчез"})
        # birds
        self.shadows = [s for s in self.shadows if t - s.t_start <= s.duration + 0.3]
        ids = flies["ids"]
        if ids and self.rng.random() < cfg.shadow_rate * self.light(t) * dt:
            fid = ids[int(self.rng.integers(len(ids)))]
            if not any(s.target == fid for s in self.shadows):
                self.shadows.append(Shadow(fid, t, direction=float(self.rng.uniform(-np.pi, np.pi))))
        self._spider(t, dt, flies)
        self._centipede(t, dt, flies)

    def _spider(self, t, dt, flies):
        s = self.spider
        ids = flies["ids"]
        if s.target is not None and s.target not in ids:
            s.target = None
        if s.target is None:
            stuck = [fid for fid, st in zip(ids, flies["stuck"]) if st]
            if stuck:
                s.target = stuck[0]
        if s.target is not None:
            i = ids.index(s.target)
            if not flies["stuck"][i]:          # it tore free
                s.target = None
                return
            tx, ty = flies["x"][i], flies["y"][i]
        else:
            tx, ty = s.home_x, s.home_y
        d = np.hypot(tx - s.x, ty - s.y)
        if d > 0.5:
            s.heading = float(np.arctan2(ty - s.y, tx - s.x))
            step = min(d, s.speed * dt)
            s.x += step * np.cos(s.heading)
            s.y += step * np.sin(s.heading)

    def _centipede(self, t, dt, flies):
        c, cfg = self.centipede, self.cfg
        ids, fx, fy = flies["ids"], flies["x"], flies["y"]
        if c.target is not None and c.target not in ids:
            c.target = None
        if c.target is not None:
            i = ids.index(c.target)
            if flies["airborne"][i] or np.hypot(fx[i] - c.x, fy[i] - c.y) > c.see_range * 1.3:
                self.log.append({"t": round(t, 2), "kind": "centipede_lost", "x": round(c.x), "y": round(c.y),
                                 "text": f"сороконожка упустила №{c.target}"})
                c.target = None
        if c.target is None and len(ids):
            dx, dy = np.asarray(fx) - c.x, np.asarray(fy) - c.y
            dist = np.hypot(dx, dy)
            rel = np.abs(np.angle(np.exp(1j * (np.arctan2(dy, dx) - c.heading))))
            seen = ((dist < c.see_range) & (rel < c.see_half_angle)) | ((dist < c.feel_range) & np.asarray(flies["moving"]))
            seen &= ~np.asarray(flies["airborne"])
            if seen.any():
                j = int(np.argmin(np.where(seen, dist, np.inf)))
                c.target = ids[j]
                self.log.append({"t": round(t, 2), "kind": "centipede_hunt", "x": round(c.x), "y": round(c.y),
                                 "text": f"сороконожка заметила №{c.target}"})
        if c.target is not None:
            i = ids.index(c.target)
            want = np.arctan2(fy[i] - c.y, fx[i] - c.x)
            speed = c.hunt_speed
        else:
            want = c.heading + self.rng.normal(0, 1.2) * np.sqrt(dt)
            speed = c.wander_speed
        turn = np.angle(np.exp(1j * (want - c.heading)))
        c.heading += float(np.clip(turn, -2.5 * dt, 2.5 * dt))
        nx, ny = c.x + speed * np.cos(c.heading) * dt, c.y + speed * np.sin(c.heading) * dt
        if nx < 10 or nx > cfg.width - 10 or ny < 10 or ny > cfg.height - 10 or self.blocked(nx, ny, pad=4):
            c.heading += float(self.rng.uniform(1.5, 3.0))
        else:
            c.x, c.y = float(nx), float(ny)
        c.body.insert(0, (round(c.x, 1), round(c.y, 1)))
        del c.body[360:]

    def apply_commands(self, path, t: float, done: int) -> int:
        """Apply viewer commands appended to a JSONL file; returns lines processed."""
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return done
        for line in lines[done:]:
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError:
                continue
            if cmd.get("type") == "fruit":
                x = float(np.clip(cmd.get("x", 0), 8, self.cfg.width - 8))
                y = float(np.clip(cmd.get("y", 0), 8, self.cfg.height - 8))
                if not self.blocked(x, y):
                    size = float(np.clip(cmd.get("size", 1.0), 0.3, 2.5))
                    self.drop_fruit(t, x, y, sugar=self.cfg.fruit_sugar * size, by_viewer=True)
        return len(lines)

    def snapshot(self, t: float) -> dict:
        s, c = self.spider, self.centipede
        return {
            "t": round(t, 3), "light": round(self.light(t), 3), "wind": [round(float(v), 2) for v in self.wind],
            "fruits": [[f.id, round(f.x, 1), round(f.y, 1), round(f.sugar), round(f.freshness(t), 2), int(f.by_viewer)]
                       for f in self.fruits],
            "shadows": [[sh.target, round((t - sh.t_start) / sh.duration, 3), round(sh.direction, 2)] for sh in self.shadows],
            "spider": [round(s.x, 1), round(s.y, 1), round(s.heading, 2), s.target if s.target is not None else -1],
            "centipede": [round(c.x, 1), round(c.y, 1), round(c.heading, 2), c.target if c.target is not None else -1,
                          c.body[::12]],
        }

    def static(self) -> dict:
        return {"width": self.cfg.width, "height": self.cfg.height, "odor_sigma": self.cfg.odor_sigma,
                "day_length": self.cfg.day_length,
                "obstacles": [[o.kind, o.x, o.y, o.rx, o.ry, o.angle] for o in self.obstacles],
                "web": [self.spider.home_x, self.spider.home_y, self.spider.web_r]}
