"""2D forest-floor world seen from above.

Units: millimetres and seconds. A fly is ~3 mm long and walks ~10-20 mm/s.
Everything here is the world model (scripted); only the flies have brains.

Food ecology (every food item has a cause and a finite amount):
- Apples appear only when a viewer drops them (commands file). They ferment
  over minutes: odor shifts from "fruit" to "vinegar".
- Predators leave droppings some time after eating a fly.
- Flies that die of hunger or old age leave a body; after a delay it
  decomposes into a small food portion.
- Droppings and decomposing bodies reuse the "vinegar" odor channel
  (fermenting / decaying matter) instead of a new odorant: fewer invented
  receptor profiles; this is a modelling choice.
- Every item has an amount that shrinks while flies and ants eat it; its
  size, odor strength and taste follow the remaining amount. Leftovers rot
  (fruit), dry out (droppings) or decay (bodies) and disappear.
- Centipedes that die (starvation or age) leave a large body: food scales
  with body size (fly body small, centipede body large, droppings small).
- Flowers (scripted plants) hold a little nectar that slowly refills. Nectar
  uses the "fruit" odorant at lower strength (sweet, fruity), a choice to
  avoid inventing a new receptor profile. A fly that fed on one flower
  carries its pollen (body state, not brain); feeding on another flower
  pollinates it, and with some chance a seedling sprouts nearby later.
  Flowers wither after their lifetime; their number is bounded, and when
  very few are left wind-blown seeds sprout on their own.
- The world starts with a few old droppings and some flowers so the first
  flies are not born into an empty world (again a choice, not a measurement).

Animals (all scripted):
- Spiders build webs where flies walk often (running heatmap of fly
  positions) or near food, spend seconds building, wait, and walk to flies
  stuck on their own webs (or on webs nobody owns). Webs age: strength,
  stickiness radius and visibility fade until they disappear. Each spider
  keeps at most `max_webs` webs. The world starts with one spider; a viewer
  can release more spiders and centipedes (commands file, capped).
- Centipedes roam and hunt flies they see (cone) or feel (vibration). They
  age and die; new ones walk in from the map edge (1-2 alive on average).
- Both predators have energy: they hunt/build only when hungry, and when
  starving they slow down and rest. Spiders never die.
- Ants from a nest search for food, eat, carry portions home and return to
  food a nestmate found (simple recruitment). They do not harm flies.
- Birds dive at flies in daylight. Wind slowly changes; odor plumes are
  stretched downwind. Water and stones block walking but not flying.

Seasons (on top of day and night): a year (default 75 min) runs spring,
summer, autumn, winter. Temperature (deg C) = annual cycle + daily cycle +
microclimate: leaf-litter shelters and the ground next to stones stay a few
degrees warmer when it is cold, the open ground is colder (analytic, no
grids). Day length follows the season. Cold slows decomposition and
fermentation (Q10 = 2), stops flowers from growing, refilling and setting
seed and makes them wither faster; predators and ants hibernate below their
threshold temperatures; birds attack less. Fly physiology does not use
temperature yet: `temperature()` is the hook (the frame carries each fly's
local temperature), cold effects on flies are left for later experiments.
"""
import json
from dataclasses import dataclass, field

import numpy as np

from . import mapgen

KINDS = ("fruit", "dropping", "corpse", "carcass", "flower")   # carcass: a dead centipede


@dataclass
class Food:
    id: int
    kind: str                   # see KINDS
    x: float
    y: float
    amount: float               # sugar-equivalent units left (a fly eats ~5/s)
    amount0: float
    born: float
    delay: float                # progress (s at 15 C) until it is food: a body decomposes first
    life: float                 # s: fruit rot time, flower lifetime; droppings/bodies: time until gone after ready
    r0: float                   # mm, radius when whole
    by_viewer: bool = False
    source: int = -1            # fly id for a body, -1 otherwise
    announced: bool = False     # ants found it / a body became food (for the feed)
    ant_seen: bool = False
    hue: float = 0.0            # flowers: petal colour for the viewer
    prog: float = 0.0           # s of fermentation/decomposition/ageing; runs faster when warm
    grow: float = 1.0           # flowers: 0 seedling .. 1 open

    def freshness(self, t: float = None) -> float:
        if self.kind != "fruit":
            return 0.0
        return float(np.clip(1 - self.prog / self.life, 0, 1))

    def decay(self, t: float = None) -> float:
        """0 fresh .. 1 gone."""
        if self.kind == "flower":
            return float(np.clip(self.prog / self.life, 0, 1))
        if self.kind == "fruit":
            return float(np.clip(self.prog / (self.life * 1.6), 0, 1))
        return float(np.clip((self.prog - self.delay) / self.life, 0, 1))

    def ready(self, t: float = None) -> bool:
        return self.grow >= 1.0 if self.kind == "flower" else self.prog >= self.delay

    def ready_frac(self) -> float:
        if self.kind == "flower":
            return self.grow
        return float(np.clip(self.prog / self.delay, 0, 1)) if self.delay > 0 else 1.0

    def radius(self) -> float:
        if self.kind == "flower":
            return self.r0
        return self.r0 * (0.5 + 0.5 * np.sqrt(max(self.amount, 0) / self.amount0))


@dataclass
class Obstacle:
    kind: str          # "water" | "stone"
    x: float
    y: float
    rx: float
    ry: float
    angle: float = 0.0

    def inside(self, px, py, pad: float = 0.0):
        return self.rnorm(px, py, pad) < 1

    def rnorm(self, px, py, pad: float = 0.0):
        """Elliptic distance: 1 on the edge."""
        c, s = np.cos(-self.angle), np.sin(-self.angle)
        dx, dy = px - self.x, py - self.y
        u, v = dx * c - dy * s, dx * s + dy * c
        return (u / (self.rx + pad)) ** 2 + (v / (self.ry + pad)) ** 2


@dataclass
class Shadow:
    """A bird diving at a fly: angular size grows until it arrives."""
    target: int
    t_start: float
    duration: float = 1.1
    direction: float = 0.0
    resolved: bool = False


@dataclass
class Web:
    id: int
    x: float
    y: float
    r: float                   # mm, full radius
    started: float
    build: float = 0.0         # 0..1 while the spider builds it
    done_at: float | None = None
    owner: int = -1            # id of the spider that built it
    strength: float = 1.0      # 1 new .. 0 gone; drops with age and when flies tear free

    def radius(self) -> float:
        if self.done_at is None:
            return self.r * self.build
        return self.r * (0.5 + 0.5 * self.strength)

    def sticky(self) -> bool:
        return self.done_at is not None or self.build >= 0.25


@dataclass
class Spider:
    id: int
    x: float
    y: float
    heading: float = 0.0
    target: int | None = None
    speed: float = 6.0
    rush_speed: float = 30.0   # mm/s dash along its web to a stuck fly
    energy: float = 0.5
    starved_for: float = 0.0
    state: str = "wait"        # wait | travel | build | rest
    site: tuple | None = None
    web: int | None = None     # web being built
    timer: float = 0.0
    detour: float = 1.0
    last_build: float = -1e9
    starving: bool = False
    hibernating: bool = False


@dataclass
class Centipede:
    id: int
    x: float
    y: float
    heading: float
    lifespan: float = 2000.0
    size: float = 1.0          # body scale; food of its body scales with it
    age: float = 0.0
    starved_for: float = 0.0
    target: int | None = None
    body: list = field(default_factory=list)   # trailing points for drawing
    wander_speed: float = 10.0
    hunt_speed: float = 24.0
    see_range: float = 45.0
    see_half_angle: float = 1.0
    feel_range: float = 12.0
    energy: float = 0.5
    rest: float = 0.0          # s left of a rest bout
    starving: bool = False
    hibernating: bool = False


@dataclass
class ArenaConfig:
    width: float = 600.0
    height: float = 400.0
    # map: "generated" (seeded, with one special biome, see mapgen.py) or "classic" (the old fixed layout)
    map: str = "generated"
    map_seed: int | None = None             # None: the world seed
    biome: str | None = None                # force "orchard" | "marsh" | "rocky" | "meadow"
    fruit_sugar: float = 250.0       # a viewer apple of size 1
    odor_ref: float = 120.0          # amount at which a source smells at full strength
    odor_sigma: float = 30.0
    shadow_rate: float = 1 / 45.0    # bird attacks per second on the whole world (daylight)
    day_length: float = 600.0
    # seasons and temperature
    year_length: float = 4500.0             # s of world time per year (75 min)
    year_start: float = 0.30                # year phase at t=0: 0 spring, .25 summer, .5 autumn, .75 winter
    temp_mean: float = 13.0                 # deg C
    temp_season_amp: float = 12.0           # midsummer mean + amp, midwinter mean - amp
    temp_day_amp: float = 3.0
    shelter_warmth: float = 4.0             # deg C warmer in leaf litter when it is cold
    stone_warmth: float = 2.5               # next to stones when it is cold
    open_chill: float = 1.5                 # open ground when it is cold
    q10: float = 2.0                        # decomposition/fermentation rate x2 per 10 deg C (ref 15)
    flower_min_temp: float = 5.0
    flower_cold_wither: float = 2.5         # flowers age this much faster below flower_min_temp
    spider_min_temp: float = 8.0            # hibernates below
    centipede_min_temp: float = 6.0
    ant_min_temp: float = 7.0
    hibernation_metabolism: float = 0.2     # fraction of normal energy use while hibernating
    # food from the ecology
    initial_droppings: int = 5
    dropping_food: float = 70.0
    dropping_delay: tuple = (60.0, 120.0)   # s after a meal
    dropping_life: float = 420.0            # s until dried out
    corpse_food: float = 35.0
    corpse_delay: float = 45.0              # s until a body has decomposed enough to feed on
    corpse_life: float = 300.0
    heat_cell: float = 20.0                 # mm, heatmap resolution
    heat_tau: float = 300.0                 # s, memory of the heatmap
    # spider
    max_webs: int = 3
    web_radius: tuple = (16.0, 24.0)
    web_build_time: float = 25.0
    web_life: float = 360.0                 # s after completion until it has decayed away
    web_cooldown: float = 45.0              # s between starting webs
    spider_hunger_s: float = 600.0          # full -> empty energy
    spider_meal: float = 0.6
    spider_starve_death: float = 400.0      # s at zero energy until a spider dies
    spider_body_food: float = 40.0
    spider_refill: tuple = (120.0, 300.0)   # s until a newcomer when no spider is alive
    spider_hungry: float = 0.75             # builds webs below this energy
    max_spiders: int = 4                    # cap for viewer-released spiders (all spiders count)
    max_centipedes_total: int = 4           # cap for viewer-released centipedes (all centipedes count)
    # centipede
    centipede_hunger_s: float = 900.0
    centipede_meal: float = 0.5
    centipede_hungry: float = 0.45          # hunts below this energy
    starving: float = 0.15                  # predators slow down and rest below this
    centipede_lifespan: tuple = (1500.0, 2700.0)
    centipede_starve_death: float = 150.0   # s at zero energy until it dies
    centipede_body_food: float = 300.0      # x body size
    carcass_delay: float = 60.0
    carcass_life: float = 600.0
    max_centipedes: int = 2
    centipede_arrival: float = 1 / 1500.0   # per s, while one is alive and there is room
    centipede_refill: tuple = (60.0, 180.0) # s until a newcomer when none is alive
    # apple trees: drop apples under the crown from midsummer to mid-autumn
    tree_crown: float = 45.0                # mm, radius where apples land
    tree_drop_rate: float = 1 / 100.0       # apples per s per tree while fruiting
    tree_season: tuple = (0.32, 0.68)       # year phase window
    tree_apple_sugar: float = 150.0
    # flowers
    initial_flowers: int = 12
    max_flowers: int = 18
    nectar_max: float = 30.0
    nectar_refill: float = 0.15             # units/s (empty -> full in ~3 min)
    flower_odor: float = 0.4                # fraction of fruit-odor strength
    flower_life: tuple = (1500.0, 3000.0)
    flower_grow: float = 60.0               # s a seedling needs to open
    seed_chance: float = 0.35               # per pollination
    seed_delay: tuple = (40.0, 90.0)        # s until the seedling sprouts
    wild_seed_rate: float = 1 / 200.0       # per s, only while fewer than 6 flowers
    pollen_life: float = 900.0              # s pollen stays on a fly
    # ants
    n_ants: int = 3
    ant_speed: float = 16.0
    ant_carry_speed: float = 11.0
    ant_sense: float = 45.0                 # mm, finds food by smell/sight within this range
    ant_bite: float = 4.0                   # units carried per trip
    ant_eat_rate: float = 3.0               # units/s while taking a portion
    ant_search: float = 90.0                # s of searching before returning home empty
    ant_nest_rest: float = 6.0


SEASONS = ("spring", "summer", "autumn", "winter")
ANT_SEARCH, ANT_TO_FOOD, ANT_EAT, ANT_HOME, ANT_NEST = range(5)


@dataclass
class Arena:
    cfg: ArenaConfig
    rng: np.random.Generator
    food: list = field(default_factory=list)
    obstacles: list = field(default_factory=list)
    shelters: list = field(default_factory=list)       # leaf-litter patches (walkable, warmer in cold)
    shadows: list = field(default_factory=list)
    webs: list = field(default_factory=list)
    spiders: list = field(default_factory=list)
    next_spider_id: int = 0
    centipedes: list = field(default_factory=list)
    next_centipede_id: int = 0
    no_centipede_since: float = 0.0
    pending_flowers: list = field(default_factory=list)     # (t_due, x, y)
    wind: np.ndarray = field(default_factory=lambda: np.zeros(2))
    ice: float = 0.0                                    # pond ice 0..1.2, frozen at >= 1
    next_food_id: int = 0
    next_web_id: int = 0
    log: list = field(default_factory=list)     # world events for the feed
    pending_droppings: list = field(default_factory=list)   # (t_due, predator object)
    heat: np.ndarray = None
    nest: tuple = (0.0, 0.0)
    pending_flies: list = field(default_factory=list)  # viewer-released flies (x, y) for Life to place
    trees: list = field(default_factory=list)          # apple trees (x, y); walkable, drop apples
    biomes: list = field(default_factory=list)         # special biome areas {kind, x, y, r} (mapgen)
    map_name: str = ""
    map_seed: int = 0
    den: tuple | None = None                           # centipede den (orchard maps): newcomers may come out of it
    nest_food: float = 0.0
    nest_known: int = -1        # food id a returning ant reported
    ants: dict = None
    counters: dict = field(default_factory=dict)
    _food_cache: tuple = (None, None)
    _clim: tuple = None
    _obs: tuple = None
    _version: int = 0

    @staticmethod
    def create(cfg: ArenaConfig, seed: int) -> "Arena":
        a = Arena(cfg, np.random.default_rng(seed))
        W, H = cfg.width, cfg.height
        a.map_seed = int(seed if cfg.map_seed is None else cfg.map_seed)
        if cfg.map == "classic":
            lay = mapgen.classic(cfg)
        else:   # own rng stream: the map does not shift the world's random sequence
            lay = mapgen.generate(cfg, np.random.default_rng([a.map_seed, 0x6D6170]))
        a.obstacles, a.shelters, a.trees, a.nest = lay.obstacles, lay.shelters, lay.trees, lay.nest
        a.biomes, a.map_name, a.den = lay.biomes, lay.name, lay.den
        a.heat = np.zeros((int(np.ceil(H / cfg.heat_cell)), int(np.ceil(W / cfg.heat_cell))))
        first = a.add_spider(*lay.spider_start, last_build=0.0)    # first new web after the heatmap has some data
        a.spawn_centipede(0.0, *lay.centipede_start, announce=False)
        n = cfg.n_ants
        a.ants = {"x": np.full(n, a.nest[0]), "y": np.full(n, a.nest[1]), "h": a.rng.uniform(-np.pi, np.pi, n),
                  "state": np.full(n, ANT_NEST), "target": np.full(n, -1), "carry": np.zeros(n),
                  "timer": a.rng.uniform(0, 20, n), "detour": np.ones(n)}
        a.counters = {"droppings": 0, "corpse_food": 0, "webs_built": 0, "ant_trips": 0, "food_by_ants": 0.0,
                      "centipedes_died": 0, "centipedes_arrived": 0, "pollinations": 0, "flowers_grown": 0,
                      "hibernations": 0}
        # the world starts mid-ecology: a few old droppings, already partly dried
        for _ in range(cfg.initial_droppings):
            x, y = a.free_spot()
            a.add_food("dropping", 0.0, x, y, cfg.dropping_food, age=float(a.rng.uniform(0, 150)))
        for k in range(cfg.initial_flowers if lay.flowers is None else len(lay.flowers)):
            x, y = a.free_spot() if lay.flowers is None else lay.flowers[k]     # generated maps: fertile ground
            a.add_flower(0.0, x, y, age=float(a.rng.uniform(cfg.flower_grow, 600)))
        # the spider starts with one finished web at its first spot
        w = Web(a.next_web_id, first.x, first.y, float(np.mean(cfg.web_radius)), 0.0, 1.0, 0.0, owner=first.id)
        a.next_web_id += 1
        a.webs.append(w)
        return a

    # --- geometry --------------------------------------------------------------
    def _inside_any(self, px, py, pad, water_only=False, solid_only=False):
        """All obstacles at once (static geometry cached as arrays)."""
        if self._obs is None:
            O = self.obstacles
            self._obs = tuple(np.array(v, dtype=float) for v in (
                [o.x for o in O], [o.y for o in O], [np.cos(-o.angle) for o in O], [np.sin(-o.angle) for o in O],
                [o.rx for o in O], [o.ry for o in O], [o.kind == "water" for o in O]))
        ox, oy, c, s, rx, ry, water = self._obs
        px, py = np.asarray(px, dtype=float), np.asarray(py, dtype=float)
        dx, dy = px[..., None] - ox, py[..., None] - oy
        hit = ((dx * c - dy * s) / (rx + pad)) ** 2 + ((dx * s + dy * c) / (ry + pad)) ** 2 < 1
        if water_only:
            hit &= water > 0
        if solid_only:
            hit &= water == 0
        return hit.any(-1)

    def blocked(self, px, py, pad: float = 0.0):
        return self._inside_any(px, py, pad)

    def walk_blocked(self, px, py, pad: float = 0.0):
        # frozen ponds can be walked on by flies (our world rule for winter)
        return self._inside_any(px, py, pad, solid_only=self.frozen)

    @property
    def frozen(self) -> bool:
        return self.ice >= 1.0

    def in_water(self, px, py):
        return self._inside_any(px, py, 0.0, water_only=True)

    def free_spot(self, margin: float = 20.0):
        for _ in range(200):
            x = float(self.rng.uniform(margin, self.cfg.width - margin))
            y = float(self.rng.uniform(margin, self.cfg.height - margin))
            if not self.blocked(x, y, pad=8) and not self.in_web(x, y, pad=15) and \
                    np.hypot(x - self.nest[0], y - self.nest[1]) > 25:
                return x, y
        return self.cfg.width / 2, self.cfg.height / 2

    def near_free(self, x, y):
        """A walkable spot close to (x, y)."""
        for r in (0.0, 4.0, 8.0, 14.0, 22.0, 35.0):
            for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                px = float(np.clip(x + r * np.cos(a), 5, self.cfg.width - 5))
                py = float(np.clip(y + r * np.sin(a), 5, self.cfg.height - 5))
                if not self.blocked(px, py, pad=2):
                    return px, py
                if r == 0:
                    break
        return self.free_spot()

    def steer(self, x, y, want, speed, dt, detour, pad=1.0):
        """Vectorised walking toward headings `want`; sliding around obstacles and map edges."""
        W, H = self.cfg.width, self.cfg.height
        x, y, want, speed, detour = (np.atleast_1d(np.asarray(v, dtype=float)).copy() for v in (x, y, want, speed, detour))
        heading = want.copy()
        nx, ny = x.copy(), y.copy()
        todo = np.ones(len(x), dtype=bool)
        inside = self.blocked(x, y, pad)       # e.g. a spider that ended a web spiral on a stone: let it walk out
        for rot in (0.0, 0.8, 1.6, -0.8, -1.6, 2.4, -2.4):
            if not todo.any():
                break
            h = want + detour * rot
            tx, ty = x + speed * np.cos(h) * dt, y + speed * np.sin(h) * dt
            ok = todo & (tx > 3) & (tx < W - 3) & (ty > 3) & (ty < H - 3) & (inside | ~self.blocked(tx, ty, pad))
            nx[ok], ny[ok], heading[ok] = tx[ok], ty[ok], h[ok]
            if rot < 0:
                detour[ok] = -detour[ok]
            todo &= ~ok
        heading[todo] = want[todo] + np.pi / 2          # boxed in: turn and try next step
        return nx, ny, heading, detour

    # --- food -----------------------------------------------------------------
    def add_food(self, kind, t, x, y, amount, by_viewer=False, age=0.0, source=-1):
        """age: seconds of progress already done (fermentation, decomposition, flower age)."""
        cfg, i = self.cfg, self.next_food_id
        if kind == "fruit":
            f = Food(i, kind, x, y, amount, amount, t, 0.0, 480.0, 6.0 * float(np.sqrt(amount / 500.0)) + 1.5, by_viewer)
        elif kind == "dropping":
            f = Food(i, kind, x, y, amount, amount, t, 0.0, cfg.dropping_life, 3.0)
        elif kind == "corpse":
            f = Food(i, kind, x, y, amount, amount, t, cfg.corpse_delay, cfg.corpse_life, 2.5, source=source)
        elif kind == "carcass":
            f = Food(i, kind, x, y, amount, amount, t, cfg.carcass_delay, cfg.carcass_life,
                     7.0 * float(np.sqrt(amount / cfg.centipede_body_food)), source=source)
        else:
            f = Food(i, kind, x, y, amount, amount, t, 0.0, float(self.rng.uniform(*cfg.flower_life)), 4.0,
                     hue=float(self.rng.uniform(0, 360)))
            f.grow = float(min(1.0, age / cfg.flower_grow))
            f.announced = f.grow >= 1.0
        f.prog = float(age)
        self.next_food_id += 1
        self.food.append(f)
        self._version += 1
        return f

    def add_flower(self, t, x, y, age=0.0):
        return self.add_food("flower", t, x, y, self.cfg.nectar_max, age=age)

    def pollinated(self, t, flower_id, pollen_from, fly_x, fly_y):
        """A fly carrying pollen fed on another flower: maybe a seedling sprouts nearby later."""
        cfg = self.cfg
        self.counters["pollinations"] += 1
        n = sum(f.kind == "flower" for f in self.food) + len(self.pending_flowers)
        if n < cfg.max_flowers and self.rng.random() < cfg.seed_chance:
            f = next((f for f in self.food if f.id == flower_id), None)
            if f is not None and float(self.temperature(t, f.x, f.y)) >= cfg.flower_min_temp:   # no seed set in the cold
                for _ in range(10):
                    a, r = self.rng.uniform(0, 2 * np.pi), self.rng.uniform(15, 40)
                    x, y = f.x + r * np.cos(a), f.y + r * np.sin(a)
                    if 10 < x < cfg.width - 10 and 10 < y < cfg.height - 10 and not self.blocked(x, y, pad=6):
                        self.pending_flowers.append((t + float(self.rng.uniform(*cfg.seed_delay)), float(x), float(y)))
                        return True
        return False

    def drop_fruit(self, t: float, x, y, sugar=None, by_viewer=True, text="зритель положил яблоко"):
        f = self.add_food("fruit", t, float(x), float(y), float(sugar or self.cfg.fruit_sugar), by_viewer=by_viewer)
        self.log.append({"t": round(t, 2), "kind": "fruit_drop" if by_viewer else "tree_apple", "x": round(f.x), "y": round(f.y),
                         "text": text})
        return f

    def _trees(self, t, dt):
        cfg = self.cfg
        lo, hi = cfg.tree_season
        if not lo <= self.year_phase(t) <= hi:
            return
        for tx, ty in self.trees:
            if self.rng.random() < cfg.tree_drop_rate * dt:
                for _ in range(20):
                    a, r = self.rng.uniform(0, 2 * np.pi), cfg.tree_crown * np.sqrt(self.rng.random())
                    x, y = tx + r * np.cos(a), ty + r * np.sin(a)
                    if 8 < x < cfg.width - 8 and 8 < y < cfg.height - 8 and not self.blocked(x, y, pad=4):
                        self.drop_fruit(t, x, y, sugar=cfg.tree_apple_sugar * self.rng.uniform(0.7, 1.2), by_viewer=False,
                                        text="с яблони упало яблоко")
                        break

    def food_arrays(self, t: float) -> dict:
        """Per-step arrays over food items (cached for the step)."""
        key = (t, self._version)
        if self._food_cache[0] == key:
            return self._food_cache[1]
        F = self.food
        amount = np.array([f.amount for f in F], dtype=float)
        ready = np.array([f.ready(t) for f in F], dtype=bool)
        fresh = np.array([f.freshness(t) for f in F])
        decay = np.array([f.decay(t) for f in F])
        kind = np.array([KINDS.index(f.kind) for f in F], dtype=int)
        s = np.clip(amount / self.cfg.odor_ref, 0, 1) * ready
        flower = kind == 4
        scale = np.where(kind == 2, 0.8, 1.0) * np.where(kind == 0, 1.0, 1 - 0.5 * decay)
        out = {
            "x": np.array([f.x for f in F], dtype=float), "y": np.array([f.y for f in F], dtype=float),
            "r": np.array([f.radius() for f in F], dtype=float), "amount": amount, "kind": kind,
            "edible": ready & (amount > 0.5),
            "fruit": np.where(flower, np.clip(amount / self.cfg.nectar_max, 0, 1) * self.cfg.flower_odor * ready, s * fresh),
            "vinegar": np.where(flower, 0.0, s * (1 - fresh) * np.where(kind == 0, 1.2, 1.0) * scale),
            "taste": np.clip(amount / np.where(flower, 10.0, 20.0), 0, 1) * ready,
            # bitter: rotting fruit, droppings and bodies (our assignment; bitter GRNs suppress feeding in the brain)
            "bitter": np.where(kind == 0, np.clip((0.4 - fresh) / 0.4, 0, 1) * 0.9,
                               np.where(kind == 1, 0.5, np.where(flower, 0.0, 0.35))) * ready,
        }
        self._food_cache = (key, out)
        return out

    def in_litter(self, px, py) -> np.ndarray:
        """1 inside a leaf-litter patch, 0 outside."""
        px, py = np.asarray(px, dtype=float), np.asarray(py, dtype=float)
        out = np.zeros(px.shape)
        for o in self.shelters:
            c, s = np.cos(-o.angle), np.sin(-o.angle)
            dx, dy = px - o.x, py - o.y
            r = np.sqrt(((dx * c - dy * s) / o.rx) ** 2 + ((dx * s + dy * c) / o.ry) ** 2)
            out = np.maximum(out, r < 1)
        return out

    def fly_died(self, t, fid, kind, x, y):
        """Bodies of flies that died of hunger or age decompose into food."""
        if kind in ("starved", "old"):
            px, py = self.near_free(x, y)
            self.add_food("corpse", t, px, py, self.cfg.corpse_food, source=fid)

    def predator_ate(self, p, t: float):
        cfg = self.cfg
        p.energy = min(1.0, p.energy + (cfg.spider_meal if isinstance(p, Spider) else cfg.centipede_meal))
        self.pending_droppings.append((t + float(self.rng.uniform(*cfg.dropping_delay)), p))

    def add_spider(self, x, y, **kw) -> Spider:
        s = Spider(self.next_spider_id, float(x), float(y), **kw)
        self.next_spider_id += 1
        self.spiders.append(s)
        return s

    def spawn_centipede(self, t, x=None, y=None, announce=True):
        cfg = self.cfg
        W, H = cfg.width, cfg.height
        text = "с края карты пришла новая сороконожка"
        if x is None and self.den is not None and self.rng.random() < 0.5:     # orchard maps: out of the den
            x, y = self.near_free(*self.den)
            text = "из логова вылезла новая сороконожка"
        if x is None:
            for _ in range(50):          # walk in from a random map edge
                side = int(self.rng.integers(4))
                u = float(self.rng.uniform(0.1, 0.9))
                x, y = [(12, u * H), (W - 12, u * H), (u * W, 12), (u * W, H - 12)][side]
                if not self.blocked(x, y, pad=6):
                    break
        c = Centipede(self.next_centipede_id, float(x), float(y), float(np.arctan2(H / 2 - y, W / 2 - x)),
                      lifespan=float(self.rng.uniform(*cfg.centipede_lifespan)), size=float(self.rng.uniform(0.8, 1.2)))
        self.next_centipede_id += 1
        self.centipedes.append(c)
        if announce:
            self.counters["centipedes_arrived"] += 1
            self._event(t, "centipede_arrived", x, y, text)
        return c

    # --- seasons ------------------------------------------------------------------
    def year_phase(self, t: float) -> float:
        return float((t / self.cfg.year_length + self.cfg.year_start) % 1.0)

    def season(self, t: float) -> str:
        return SEASONS[int(self.year_phase(t) * 4) % 4]

    def _seasonal(self, t: float) -> float:
        """+1 at midsummer, -1 at midwinter."""
        return float(np.cos(2 * np.pi * (self.year_phase(t) - 0.375)))

    def light(self, t: float) -> float:
        # long bright days in summer, short dim ones in winter (light > 0.55 = day: 64% vs 36% of the cycle)
        phase = (t % self.cfg.day_length) / self.cfg.day_length
        return float(np.clip(0.55 + 0.45 * np.cos(2 * np.pi * (phase - 0.25)) + 0.2 * self._seasonal(t), 0.1, 1.0))

    def air_temperature(self, t: float) -> float:
        """Open-ground air temperature, deg C: annual + daily cycle (warmest early afternoon)."""
        cfg = self.cfg
        day = np.cos(2 * np.pi * ((t % cfg.day_length) / cfg.day_length - 0.3))
        return float(cfg.temp_mean + cfg.temp_season_amp * self._seasonal(t) + cfg.temp_day_amp * day)

    def temperature(self, t: float, px, py):
        """Local temperature at points (deg C). Hook for fly physiology; cheap analytic microclimate."""
        cfg = self.cfg
        T = self.air_temperature(t)
        cold = float(np.clip((12.0 - T) / 12.0, 0, 1))
        hot = float(np.clip((T - 20.0) / 8.0, 0, 1))
        px, py = np.asarray(px, dtype=float), np.asarray(py, dtype=float)
        if cold == 0 and hot == 0:
            return np.full(px.shape, T) if px.ndim else T
        if self._clim is None:                       # static geometry as arrays, built once
            sh, st = self.shelters, [o for o in self.obstacles if o.kind == "stone"]
            self._clim = tuple(np.array(v, dtype=float) for v in (
                [o.x for o in sh], [o.y for o in sh], [np.cos(-o.angle) for o in sh], [np.sin(-o.angle) for o in sh],
                [o.rx for o in sh], [o.ry for o in sh], [o.x for o in st], [o.y for o in st], [o.rx for o in st]))
        sx, sy, sc, ss, srx, sry, ox, oy, orad = self._clim
        qx, qy = px[..., None], py[..., None]
        dx, dy = qx - sx, qy - sy
        r = np.sqrt(((dx * sc - dy * ss) / srx) ** 2 + ((dx * ss + dy * sc) / sry) ** 2)
        litter = np.clip((1.5 - r) / 0.5, 0, 1).max(-1)                   # 1 inside a patch, 0 beyond 1.5x its radius
        stone = np.clip(1 - (np.hypot(qx - ox, qy - oy) - orad) / 12.0, 0, 1).max(-1)
        cover = np.maximum(litter, 0.7 * stone)
        out = T + cold * (cfg.shelter_warmth * litter + cfg.stone_warmth * stone * (1 - litter)) \
            - cold * cfg.open_chill * (1 - cover) - hot * 2.0 * cover       # shade is cooler in the heat
        return out if px.ndim else float(out)

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
            if self.food:   # all sources at once along a trailing axis
                fa = self.food_arrays(t)
                c = plume(fa["x"], fa["y"], fa[name], sig).sum(-1)
        elif name == "spider":
            for sp in self.spiders:
                c += plume(sp.x, sp.y, 0.8, sig * 0.7)
        elif name == "centipede":
            for cp in self.centipedes:
                c += plume(cp.x, cp.y, 1.0 * cp.size, sig * 0.6)
        return c

    # --- webs -------------------------------------------------------------------
    def in_web(self, px, py, pad: float = 0.0):
        """Id+1 of the sticky web at each point, 0 if none."""
        out = np.zeros(np.shape(px), dtype=int)
        for w in self.webs:
            if w.sticky() or pad > 0:
                hit = (np.hypot(np.asarray(px) - w.x, np.asarray(py) - w.y) < w.radius() + pad) & (out == 0)
                out = np.where(hit, w.id + 1, out)
        return out

    def web(self, wid: int) -> Web | None:
        return next((w for w in self.webs if w.id == wid), None)

    def tear_web(self, wid: int, t: float, damage: float = 0.25):
        w = self.web(wid)
        if w is not None:
            w.strength -= damage

    def _web_site(self, s: Spider):
        """Where to build: cells flies walked through lately, plus smell of food nearby."""
        cfg = self.cfg
        cs = cfg.heat_cell
        gy, gx = np.mgrid[0:self.heat.shape[0], 0:self.heat.shape[1]]
        cx, cy = (gx.ravel() + 0.5) * cs, (gy.ravel() + 0.5) * cs
        score = self.heat.ravel() / max(self.heat.max(), 1e-6)
        if self.food:
            fx = np.array([f.x for f in self.food])
            fy = np.array([f.y for f in self.food])
            fam = np.array([min(1.0, f.amount / cfg.odor_ref) for f in self.food])
            d = np.hypot(cx[:, None] - fx, cy[:, None] - fy)
            score = score + 0.6 * (fam * np.exp(-d ** 2 / (2 * 35.0 ** 2))).sum(1)
            too_close_food = (d < 14).any(1)
        else:
            too_close_food = np.zeros(len(cx), dtype=bool)
        r = cfg.web_radius[1]
        ok = ~self.blocked(cx, cy, pad=r * 0.6) & (cx > r) & (cx < cfg.width - r) & (cy > r) & (cy < cfg.height - r)
        ok &= ~too_close_food & (np.hypot(cx - self.nest[0], cy - self.nest[1]) > 40)
        for w in self.webs:
            ok &= np.hypot(cx - w.x, cy - w.y) > w.r + r + 6
        if not ok.any():
            return None
        # a spider walks ~6 mm/s: far sites cost it
        score = score - 0.4 * np.hypot(cx - s.x, cy - s.y) / max(cfg.width, cfg.height)
        score = np.where(ok, score + 0.05 * self.rng.random(len(score)), -np.inf)
        top = np.argsort(-score)[:5]
        top = top[np.isfinite(score[top])]
        j = int(self.rng.choice(top))
        jitter = self.rng.uniform(-cs / 3, cs / 3, 2)
        x, y = float(cx[j] + jitter[0]), float(cy[j] + jitter[1])
        if self.blocked(x, y, pad=r * 0.6):
            x, y = float(cx[j]), float(cy[j])
        return x, y

    # --- world update -----------------------------------------------------------
    def update(self, t: float, dt: float, flies: dict):
        """flies: {"ids", "x", "y", "stuck" (web id+1 or 0), "airborne", "moving"} arrays of living flies."""
        cfg = self.cfg
        # wind drifts slowly
        self.wind += self.rng.normal(0, 0.03, 2) * np.sqrt(dt) - self.wind * 0.02 * dt
        self.wind = np.clip(self.wind, -0.8, 0.8)
        # heatmap of where flies walk
        self.heat *= np.exp(-dt / cfg.heat_tau)
        walk = ~np.asarray(flies["airborne"], dtype=bool)
        if walk.any():
            gx = np.clip((np.asarray(flies["x"])[walk] / cfg.heat_cell).astype(int), 0, self.heat.shape[1] - 1)
            gy = np.clip((np.asarray(flies["y"])[walk] / cfg.heat_cell).astype(int), 0, self.heat.shape[0] - 1)
            np.add.at(self.heat, (gy, gx), dt)
        self._food_update(t, dt)
        self._trees(t, dt)
        # ponds freeze after a cold spell (< 3 C air) and thaw only above 5 C (ice 0..1.2, frozen at >= 1)
        was = self.frozen
        air = self.air_temperature(t)
        self.ice = float(np.clip(self.ice + ((3.0 - air) / 60.0 if air < 3.0 else -(air - 5.0) / 30.0 if air > 5.0 else 0.0) * dt, 0.0, 1.2))
        if self.frozen != was:
            self.log.append({"t": round(t, 2), "kind": "ice_on" if self.frozen else "ice_off", "x": None, "y": None,
                             "text": "пруды замёрзли — по льду можно ходить" if self.frozen else "лёд растаял"})
        # birds
        self.shadows = [s for s in self.shadows if t - s.t_start <= s.duration + 0.3]
        ids = flies["ids"]
        bird_season = float(np.clip(self.air_temperature(t) / 20.0, 0.05, 1.0))    # fewer birds in the cold
        if ids and self.rng.random() < cfg.shadow_rate * self.light(t) * bird_season * dt:
            # a bird only spots flies in the open: not under a tree crown, not in leaf litter
            fx, fy = np.asarray(flies["x"], dtype=float), np.asarray(flies["y"], dtype=float)
            covered = self.in_litter(fx, fy) > 0
            for tx, ty in self.trees:
                covered |= np.hypot(fx - tx, fy - ty) < cfg.tree_crown
            open_ids = [f for f, c in zip(ids, covered) if not c]
            fid = open_ids[int(self.rng.integers(len(open_ids)))] if open_ids else None
            if fid is not None and not any(s.target == fid for s in self.shadows):
                self.shadows.append(Shadow(fid, t, direction=float(self.rng.uniform(-np.pi, np.pi))))
        self._webs(t, dt)
        taken = {sp.target for sp in self.spiders} - {None}
        for sp in self.spiders:
            self._spider(sp, t, dt, flies, taken)
        for c in list(self.centipedes):
            self._centipede(c, t, dt, flies)
        self._centipede_population(t, dt)
        self._spider_population(t, dt)
        self._ants(t, dt)

    def _event(self, t, kind, x, y, text):
        self.log.append({"t": round(t, 2), "kind": kind, "x": round(float(x)), "y": round(float(y)), "text": text})

    def _food_update(self, t, dt):
        cfg = self.cfg
        for item in [q for q in self.pending_droppings if q[0] <= t]:
            self.pending_droppings.remove(item)
            p = item[1]
            x, y = self.near_free(p.x, p.y)
            self.add_food("dropping", t, x, y, cfg.dropping_food)
            self.counters["droppings"] += 1
            self._event(t, "dropping", x, y, "паук оставил помёт — еда" if isinstance(p, Spider) else "сороконожка оставила помёт — еда")
        # flowers: seedlings sprout, nectar refills, wind-blown seeds when few are left
        for item in [q for q in self.pending_flowers if q[0] <= t]:
            self.pending_flowers.remove(item)
            self.add_flower(t, item[1], item[2])
            self._event(t, "flower_sprout", item[1], item[2], "из пыльцы проклюнулся росток")
        n_flowers = 0
        if self.food:
            temp = self.temperature(t, np.array([f.x for f in self.food]), np.array([f.y for f in self.food]))
            rate = np.clip(cfg.q10 ** ((temp - 15.0) / 10.0), 0.1, 1.6)
        for k, f in enumerate(self.food):
            if f.kind != "flower":
                f.prog += dt * rate[k]
                continue
            n_flowers += 1
            warm = temp[k] >= cfg.flower_min_temp
            f.prog += dt * (1.0 if warm else cfg.flower_cold_wither)
            if not warm:
                continue
            if f.grow < 1.0:
                f.grow = min(1.0, f.grow + dt / cfg.flower_grow)
            else:
                f.amount = min(f.amount0, f.amount + cfg.nectar_refill * dt)
                if not f.announced:
                    f.announced = True
                    self.counters["flowers_grown"] += 1
                    self._event(t, "flower_grown", f.x, f.y, "вырос новый цветок")
        if n_flowers + len(self.pending_flowers) < 6 and self.air_temperature(t) >= cfg.flower_min_temp \
                and self.rng.random() < cfg.wild_seed_rate * dt:
            x, y = self.free_spot()
            self.add_flower(t, x, y)
            self._event(t, "flower_sprout", x, y, "ветер занёс семя — проклюнулся росток")
        for f in list(self.food):
            if f.kind in ("corpse", "carcass") and not f.announced and f.ready(t) and f.amount > 0.5:
                f.announced = True
                self.counters["corpse_food"] += 1
                self._event(t, "corpse_food", f.x, f.y, f"тело мухи №{f.source} разложилось — стало едой" if f.kind == "corpse"
                            else "тело сороконожки разложилось — много еды")
            eaten = f.amount <= 0.5 and f.kind != "flower"
            if eaten or f.decay(t) >= 1:
                self.food.remove(f)
                self._version += 1
                if f.kind == "fruit":
                    self._event(t, "fruit_gone", f.x, f.y, "яблоко съедено" if eaten else "яблоко сгнило и исчезло")
                elif f.kind == "dropping":
                    self._event(t, "food_gone", f.x, f.y, "помёт съеден" if eaten else "помёт высох")
                elif f.kind == "flower":
                    self._event(t, "flower_gone", f.x, f.y, "цветок завял" if f.grow >= 1 else "росток погиб")
                else:
                    self._event(t, "food_gone", f.x, f.y, "останки съедены" if eaten else "останки истлели")

    def _webs(self, t, dt):
        cfg = self.cfg
        for w in list(self.webs):
            if w.done_at is not None:
                w.strength -= dt / cfg.web_life
            if w.strength <= 0:
                self.webs.remove(w)
                self._event(t, "web_gone", w.x, w.y, "старая паутина разрушилась")

    def _spider(self, s, t, dt, flies, taken):
        """taken: fly ids spiders are already walking to (kept up to date here)."""
        cfg = self.cfg
        ids = flies["ids"]
        if self._hibernate(s, t, cfg.spider_min_temp, "паук"):
            s.energy = max(0.0, s.energy - cfg.hibernation_metabolism * dt / cfg.spider_hunger_s)
            taken.discard(s.target)
            s.target = None
            return                                  # does not move, hunt or build (a web in progress waits)
        s.energy = max(0.0, s.energy - dt / cfg.spider_hunger_s)
        s.starved_for = s.starved_for + dt if s.energy <= 0 else 0.0
        starving = s.energy < cfg.starving
        if starving and not s.starving:
            self._event(t, "spider_starving", s.x, s.y, "паук ослаб от голода — медлит и отдыхает")
        s.starving = starving
        speed = s.speed * (0.5 if starving else 1.0)
        if s.target is not None and (s.target not in ids or not flies["stuck"][ids.index(s.target)]):
            taken.discard(s.target)
            s.target = None                         # eaten by someone else, or it tore free
        if s.target is None and len(ids):
            # flies stuck on its own webs, or on webs whose owner is gone, not claimed by another spider
            alive = {sp.id for sp in self.spiders}
            mine = [w.id + 1 for w in self.webs if w.owner == s.id or w.owner not in alive]
            stuck = np.flatnonzero(np.isin(np.asarray(flies["stuck"]), mine) & ~np.isin(ids, list(taken)))
            if len(stuck):
                d = np.hypot(np.asarray(flies["x"])[stuck] - s.x, np.asarray(flies["y"])[stuck] - s.y)
                s.target = ids[int(stuck[np.argmin(d)])]
                taken.add(s.target)
                if s.state == "build":
                    self._finish_build(s, t, interrupted=True)
                s.state = "wait"
        if s.target is not None:
            i = ids.index(s.target)
            self._spider_walk(s, flies["x"][i], flies["y"][i], s.rush_speed * (0.6 if starving else 1.0), dt)
            return
        if s.state == "rest":
            s.timer -= dt
            if s.timer <= 0:
                s.state = "wait"
            return
        if s.state == "travel":
            if self._spider_walk(s, *s.site, speed, dt) < 1.0:
                w = Web(self.next_web_id, s.site[0], s.site[1], float(self.rng.uniform(*cfg.web_radius)), t, owner=s.id)
                self.next_web_id += 1
                self.webs.append(w)
                s.web, s.state, s.timer = w.id, "build", 0.0
                self._event(t, "web_start", w.x, w.y, "паук начал плести паутину")
            return
        if s.state == "build":
            w = self.web(s.web)
            if w is None:
                s.state = "wait"
                return
            w.build = min(1.0, w.build + dt / cfg.web_build_time * (0.5 if starving else 1.0))
            s.timer += dt * 2.2                      # circling the hub, spiral outwards
            rr = w.r * max(w.build, 0.1)
            nx, ny = w.x + rr * np.cos(s.timer), w.y + rr * np.sin(s.timer)
            s.heading = float(np.arctan2(ny - s.y, nx - s.x))
            s.x, s.y = float(nx), float(ny)
            if w.build >= 1.0:
                self._finish_build(s, t)
            return
        # wait at the hub of its newest web; hungry spiders build more webs
        own = [w for w in self.webs if w.owner == s.id]
        home = max(own, key=lambda w: w.started) if own else None
        if home is not None:
            self._spider_walk(s, home.x, home.y, speed, dt)
        if starving and self.rng.random() < dt / 30.0:
            s.state, s.timer = "rest", float(self.rng.uniform(15, 30))
            return
        if s.energy < cfg.spider_hungry and len(own) < cfg.max_webs and t - s.last_build > cfg.web_cooldown:
            site = self._web_site(s)
            s.last_build = t
            if site is not None:
                s.site, s.state = site, "travel"

    def _spider_population(self, t, dt):
        cfg = self.cfg
        for s in list(self.spiders):
            if s.starved_for >= cfg.spider_starve_death:
                self.spiders.remove(s)
                x, y = self.near_free(s.x, s.y)
                self.add_food("carcass", t, x, y, cfg.spider_body_food, source=-1)
                self._event(t, "spider_died", x, y, "паук умер от голода")
                self.no_spider_since = t
        if not self.spiders and t - getattr(self, "no_spider_since", 0.0) > self.rng.uniform(*cfg.spider_refill) \
                and self.rng.random() < dt / 10 and float(self.air_temperature(t)) >= cfg.spider_min_temp:
            for _ in range(50):
                side, u = int(self.rng.integers(4)), float(self.rng.uniform(0.1, 0.9))
                x, y = [(15, u * cfg.height), (cfg.width - 15, u * cfg.height), (u * cfg.width, 15), (u * cfg.width, cfg.height - 15)][side]
                if not self.blocked(x, y, pad=6):
                    self.add_spider(x, y, heading=float(self.rng.uniform(-np.pi, np.pi)))
                    self._event(t, "spider_arrived", x, y, "с края карты пришёл новый паук")
                    break

    def _hibernate(self, p, t, min_temp, name) -> bool:
        cold = float(self.temperature(t, p.x, p.y)) < min_temp - (0.0 if not p.hibernating else -1.0)   # 1 deg hysteresis
        if cold != p.hibernating:
            p.hibernating = cold
            if cold:
                self.counters["hibernations"] += 1
            self._event(t, "hibernate" if cold else "wake", p.x, p.y,
                        f"{name}: холодно — оцепенение, спячка" if cold else f"{name}: потеплело — проснулся")
        return cold

    def _finish_build(self, s, t, interrupted=False):
        w = self.web(s.web)
        s.web, s.state = None, "wait"
        if w is None:
            return
        if interrupted and w.build < 0.3:
            self.webs.remove(w)
            return
        if interrupted:
            w.r *= w.build
        w.build, w.done_at = 1.0, t
        self.counters["webs_built"] += 1
        self._event(t, "web_done", w.x, w.y, "паутина готова" if not interrupted else "паук бросил недоплетённую паутину")

    def _spider_walk(self, s, tx, ty, speed, dt) -> float:
        d = float(np.hypot(tx - s.x, ty - s.y))
        if d > 0.5:
            want = np.arctan2(ty - s.y, tx - s.x)
            # webs are the spider's own ground: it may cross them; water and stones it walks around
            nx, ny, h, det = self.steer(s.x, s.y, want, min(speed, d / dt), dt, s.detour, pad=0.5)
            s.x, s.y, s.heading, s.detour = float(nx[0]), float(ny[0]), float(h[0]), float(det[0])
        return d

    def _centipede_population(self, t, dt):
        cfg = self.cfg
        for c in list(self.centipedes):
            reason = "old" if c.age >= c.lifespan else "starved" if c.starved_for >= cfg.centipede_starve_death else None
            if reason:
                self.centipedes.remove(c)
                self.counters["centipedes_died"] += 1
                x, y = self.near_free(c.x, c.y)
                self.add_food("carcass", t, x, y, cfg.centipede_body_food * c.size, source=c.id)
                self._event(t, "centipede_died", x, y, "сороконожка умерла от старости" if reason == "old"
                            else "сороконожка умерла от голода")
                self.no_centipede_since = t
        if not self.centipedes:
            if t - self.no_centipede_since > self.rng.uniform(*cfg.centipede_refill) and self.rng.random() < dt / 10:
                self.spawn_centipede(t)
        elif len(self.centipedes) < cfg.max_centipedes and self.rng.random() < cfg.centipede_arrival * dt:
            self.spawn_centipede(t)

    def _centipede(self, c, t, dt, flies):
        cfg = self.cfg
        ids, fx, fy = flies["ids"], flies["x"], flies["y"]
        c.age += dt
        if self._hibernate(c, t, cfg.centipede_min_temp, "сороконожка"):
            c.energy = max(0.0, c.energy - cfg.hibernation_metabolism * dt / cfg.centipede_hunger_s)
            c.target, c.starved_for = None, 0.0
            return
        c.energy = max(0.0, c.energy - dt / cfg.centipede_hunger_s)
        c.starved_for = c.starved_for + dt if c.energy <= 0 else 0.0
        starving = c.energy < cfg.starving
        if starving and not c.starving:
            self._event(t, "centipede_starving", c.x, c.y, "сороконожка ослабла от голода — медлит и отдыхает")
        c.starving = starving
        hungry = c.energy < cfg.centipede_hungry
        if c.target is not None and c.target not in ids:
            c.target = None
        if c.target is not None:
            i = ids.index(c.target)
            if flies["airborne"][i] or np.hypot(fx[i] - c.x, fy[i] - c.y) > c.see_range * 1.3:
                self._event(t, "centipede_lost", c.x, c.y, f"сороконожка упустила №{c.target}")
                c.target = None
        if c.rest > 0:
            c.rest -= dt
            c.body.insert(0, (round(c.x, 1), round(c.y, 1)))
            del c.body[360:]
            return
        if starving and c.target is None and self.rng.random() < dt / 25.0:
            c.rest = float(self.rng.uniform(10, 20))
            return
        taken = {o.target for o in self.centipedes if o is not c}
        if c.target is None and len(ids) and hungry:
            dx, dy = np.asarray(fx) - c.x, np.asarray(fy) - c.y
            dist = np.hypot(dx, dy)
            rel = np.abs(np.angle(np.exp(1j * (np.arctan2(dy, dx) - c.heading))))
            seen = ((dist < c.see_range) & (rel < c.see_half_angle)) | ((dist < c.feel_range) & np.asarray(flies["moving"]))
            seen &= ~np.asarray(flies["airborne"]) & ~np.isin(ids, list(taken - {None}))
            if seen.any():
                j = int(np.argmin(np.where(seen, dist, np.inf)))
                c.target = ids[j]
                self._event(t, "centipede_hunt", c.x, c.y, f"сороконожка заметила №{c.target}")
        slow = 0.5 if starving else 1.0
        if c.target is not None:
            i = ids.index(c.target)
            want = np.arctan2(fy[i] - c.y, fx[i] - c.x)
            speed = c.hunt_speed * slow
        else:
            want = c.heading + self.rng.normal(0, 1.2) * np.sqrt(dt)
            speed = c.wander_speed * slow * (1.0 if hungry else 0.6)     # a fed centipede ambles
        turn = np.angle(np.exp(1j * (want - c.heading)))
        c.heading += float(np.clip(turn, -2.5 * dt, 2.5 * dt))
        nx, ny = c.x + speed * np.cos(c.heading) * dt, c.y + speed * np.sin(c.heading) * dt
        if nx < 10 or nx > cfg.width - 10 or ny < 10 or ny > cfg.height - 10 or self.blocked(nx, ny, pad=4):
            c.heading += float(self.rng.uniform(1.5, 3.0))
        else:
            c.x, c.y = float(nx), float(ny)
        c.body.insert(0, (round(c.x, 1), round(c.y, 1)))
        del c.body[360:]

    def _ants(self, t, dt):
        A, cfg = self.ants, self.cfg
        n = len(A["x"])
        if not n:
            return
        fa = self.food_arrays(t)
        fid = np.array([f.id for f in self.food], dtype=int)
        by_id = {f.id: k for k, f in enumerate(self.food)}
        st = A["state"]
        cold = self.air_temperature(t) < cfg.ant_min_temp       # the colony stays in the nest
        if cold:
            st[(st == ANT_SEARCH) | (st == ANT_TO_FOOD) | (st == ANT_EAT)] = ANT_HOME
        # nearest ground food for every searching ant at once
        search = np.flatnonzero(st == ANT_SEARCH)
        found = {}
        if len(search) and len(fid):
            d = np.hypot(fa["x"][None] - A["x"][search, None], fa["y"][None] - A["y"][search, None])
            d = np.where(fa["edible"] & (fa["kind"] != 4), d, np.inf)     # ground foragers: no nectar
            k = np.argmin(d, 1)
            for a, kk, dd in zip(search, k, d[np.arange(len(search)), k]):
                if dd < cfg.ant_sense:
                    found[a] = fid[kk]
        # rare transitions in a small loop (a handful of ants)
        for a in range(n):
            s = st[a]
            if s == ANT_NEST:
                if cold:
                    continue
                A["timer"][a] -= dt
                if A["timer"][a] <= 0:
                    k = by_id.get(self.nest_known)
                    if k is not None and fa["edible"][k]:
                        st[a], A["target"][a] = ANT_TO_FOOD, self.nest_known
                    else:
                        self.nest_known = -1
                        st[a], A["timer"][a] = ANT_SEARCH, cfg.ant_search
                        A["h"][a] = self.rng.uniform(-np.pi, np.pi)
            elif s == ANT_SEARCH:
                A["timer"][a] -= dt
                if a in found:
                    st[a], A["target"][a] = ANT_TO_FOOD, found[a]
                    continue
                if A["timer"][a] <= 0:
                    st[a] = ANT_HOME
            elif s in (ANT_TO_FOOD, ANT_EAT):
                k = by_id.get(int(A["target"][a]))
                if k is None or not fa["edible"][k]:
                    st[a], A["target"][a] = (ANT_HOME if A["carry"][a] > 0 else ANT_SEARCH), -1
                    A["timer"][a] = cfg.ant_search / 2
                    continue
                f = self.food[k]
                if s == ANT_TO_FOOD:
                    if np.hypot(f.x - A["x"][a], f.y - A["y"][a]) < fa["r"][k] + 1.0:
                        st[a] = ANT_EAT
                        if not f.ant_seen:
                            f.ant_seen = True
                            self._event(t, "ant_food", f.x, f.y, "муравьи нашли еду")
                else:
                    bite = min(cfg.ant_eat_rate * dt, f.amount)
                    f.amount -= bite
                    A["carry"][a] += bite
                    self.counters["food_by_ants"] += bite
                    if A["carry"][a] >= cfg.ant_bite or f.amount <= 0.5:
                        st[a] = ANT_HOME
                        self.nest_known = f.id if f.amount > cfg.ant_bite else -1
            elif s == ANT_HOME:
                if np.hypot(self.nest[0] - A["x"][a], self.nest[1] - A["y"][a]) < 3.0:
                    st[a], A["timer"][a] = ANT_NEST, cfg.ant_nest_rest
                    if A["carry"][a] > 0:
                        self.counters["ant_trips"] += 1
                    self.nest_food += A["carry"][a]
                    A["carry"][a] = 0.0
                    A["x"][a], A["y"][a] = self.nest
        # vectorised walking
        move = (st == ANT_SEARCH) | (st == ANT_TO_FOOD) | (st == ANT_HOME)
        if not move.any():
            return
        want = A["h"] + self.rng.normal(0, 1.5, n) * np.sqrt(dt)
        tgt = st == ANT_TO_FOOD
        if tgt.any():
            k = np.array([by_id.get(int(g), 0) for g in A["target"][tgt]])
            want[tgt] = np.arctan2(fa["y"][k] - A["y"][tgt], fa["x"][k] - A["x"][tgt])
        home = st == ANT_HOME
        want[home] = np.arctan2(self.nest[1] - A["y"][home], self.nest[0] - A["x"][home]) + self.rng.normal(0, 0.3, home.sum())
        speed = np.where(A["carry"] > 0, cfg.ant_carry_speed, cfg.ant_speed)
        m = np.flatnonzero(move)
        nx, ny, h, det = self.steer(A["x"][m], A["y"][m], want[m], speed[m], dt, A["detour"][m], pad=0.5)
        A["x"][m], A["y"][m], A["h"][m], A["detour"][m] = nx, ny, h, det

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
            kind = cmd.get("type")
            try:
                x, y = float(cmd.get("x", 0)), float(cmd.get("y", 0))
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if kind == "fruit":
                x, y = float(np.clip(x, 8, self.cfg.width - 8)), float(np.clip(y, 8, self.cfg.height - 8))
                if not self.blocked(x, y):
                    size = float(np.clip(cmd.get("size", 1.0), 0.3, 2.5))
                    self.drop_fruit(t, x, y, sugar=self.cfg.fruit_sugar * size, by_viewer=True)
            elif kind in ("centipede", "spider"):
                self.release_predator(t, kind, x, y)
            elif kind == "fly":
                x, y = float(np.clip(x, 8, self.cfg.width - 8)), float(np.clip(y, 8, self.cfg.height - 8))
                self.pending_flies.append(self.near_free(x, y))
        return len(lines)

    def release_predator(self, t, kind, x, y):
        """A viewer releases a spider or a centipede: capped, not into water or onto stones."""
        cfg = self.cfg
        x, y = float(np.clip(x, 12, cfg.width - 12)), float(np.clip(y, 12, cfg.height - 12))
        name = "паука" if kind == "spider" else "сороконожку"
        if self.blocked(x, y, pad=4):
            self._event(t, "user_blocked", x, y, f"сюда нельзя выпустить {name}: вода или камень")
            return None
        if kind == "spider":
            if len(self.spiders) >= cfg.max_spiders:
                self._event(t, "user_cap", x, y, f"пауков уже {len(self.spiders)} — больше выпустить нельзя")
                return None
            p = self.add_spider(x, y, heading=float(self.rng.uniform(-np.pi, np.pi)))
            self._event(t, "user_spider", x, y, "пользователь выпустил паука")
        else:
            if len(self.centipedes) >= cfg.max_centipedes_total:
                self._event(t, "user_cap", x, y, f"сороконожек уже {len(self.centipedes)} — больше выпустить нельзя")
                return None
            p = self.spawn_centipede(t, x, y, announce=False)
            self._event(t, "user_centipede", x, y, "пользователь выпустил сороконожку")
        return p

    def snapshot(self, t: float) -> dict:
        A = self.ants
        return {
            "t": round(t, 3), "light": round(self.light(t), 3), "wind": [round(float(v), 2) for v in self.wind],
            "season": self.season(t), "year_phase": round(self.year_phase(t), 4), "temp": round(self.air_temperature(t), 1),
            "ice": round(self.ice, 2),
            # fruits: [id, x, y, amount, freshness, by_viewer, amount_whole]
            "fruits": [[f.id, round(f.x, 1), round(f.y, 1), round(f.amount), round(f.freshness(t), 2), int(f.by_viewer),
                        round(f.amount0)] for f in self.food if f.kind == "fruit"],
            # food: droppings and bodies [id, kind, x, y, amount, amount_whole, ready 0..1, decay 0..1, source]
            "food": [[f.id, f.kind, round(f.x, 1), round(f.y, 1), round(f.amount, 1), round(f.amount0),
                      round(f.ready_frac(), 2), round(f.decay(t), 2),
                      f.source] for f in self.food if f.kind not in ("fruit", "flower")],
            # flowers: [id, x, y, nectar, nectar_max, grown 0..1, wither 0..1, hue]
            "flowers": [[f.id, round(f.x, 1), round(f.y, 1), round(f.amount, 1), round(f.amount0),
                         round(f.grow, 2), round(f.decay(t), 2),
                         round(f.hue)] for f in self.food if f.kind == "flower"],
            "shadows": [[sh.target, round((t - sh.t_start) / sh.duration, 3), round(sh.direction, 2)] for sh in self.shadows],
            # webs: [id, x, y, radius, strength, build]
            "webs": [[w.id, round(w.x, 1), round(w.y, 1), round(w.radius(), 1), round(w.strength, 2), round(w.build, 2)]
                     for w in self.webs],
            # spiders: [x, y, heading, target, state, energy, hibernating, id]
            "spiders": [[round(s.x, 1), round(s.y, 1), round(s.heading, 2), s.target if s.target is not None else -1,
                         s.state, round(s.energy, 2), int(s.hibernating), s.id] for s in self.spiders],
            # centipedes: [id, x, y, heading, target, body points, energy, resting, size, age/lifespan, hibernating]
            "centipedes": [[c.id, round(c.x, 1), round(c.y, 1), round(c.heading, 2), c.target if c.target is not None else -1,
                            c.body[::12], round(c.energy, 2), int(c.rest > 0), round(c.size, 2), round(c.age / c.lifespan, 2),
                            int(c.hibernating)]
                           for c in self.centipedes],
            # ants: [x, y, heading, state, carrying]
            "ants": [[round(float(A["x"][a]), 1), round(float(A["y"][a]), 1), round(float(A["h"][a]), 2), int(A["state"][a]),
                      int(A["carry"][a] > 0)] for a in range(len(A["x"]))],
            "nest_food": round(self.nest_food),
        }

    def static(self) -> dict:
        return {"width": self.cfg.width, "height": self.cfg.height, "odor_sigma": self.cfg.odor_sigma,
                "day_length": self.cfg.day_length, "odor_ref": self.cfg.odor_ref,
                "obstacles": [[o.kind, o.x, o.y, o.rx, o.ry, o.angle] for o in self.obstacles],
                "nest": [round(self.nest[0], 1), round(self.nest[1], 1)], "trees": [[round(x, 1), round(y, 1)] for x, y in self.trees],
                "tree_crown": self.cfg.tree_crown, "max_webs": self.cfg.max_webs,
                "shelters": [[o.kind, o.x, o.y, o.rx, o.ry, o.angle] for o in self.shelters],
                "year_length": self.cfg.year_length, "year_start": self.cfg.year_start, "seasons": list(SEASONS),
                "map": self.cfg.map, "map_seed": self.map_seed, "map_name": self.map_name, "biomes": self.biomes,
                "den": [round(self.den[0], 1), round(self.den[1], 1)] if self.den else None}
