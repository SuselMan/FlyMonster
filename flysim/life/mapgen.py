"""Seeded map generation for the arena (world rule, ours).

Two smooth value-noise fields over the map (numpy only): moisture and fertility.
Ponds go where it is wettest, apple trees and flowers where it is fertile, the ant
nest on dry ground away from water, leaf litter in moist spots near trees, stones
anywhere. Every map also gets one special biome picked by the rng:

- orchard: a cluster of apple trees and extra flowers, but the spider starts there
  and a centipede den is next to it (much food, many predators)
- marsh:   many small ponds, humid, few stones
- rocky:   many stones, dry, a single small pond
- meadow:  many flowers

A map is accepted only if all free ground is connected (flood fill on a 5 mm grid:
ponds and stones never cut the map in two) and enough of it is free.
`classic()` returns the old fixed layout.
"""
from dataclasses import dataclass, field

import numpy as np

BIOMES = ("orchard", "marsh", "rocky", "meadow")
BIOME_RU = {"orchard": "Яблоневый сад", "marsh": "Болото", "rocky": "Каменистая пустошь", "meadow": "Цветущий луг",
            "classic": "Классическая карта"}
GRID = 5.0          # mm, walkability grid
EDGE = 25.0         # ponds keep this far from the map edge
GAP = 14.0          # mm of free ground kept between obstacles


@dataclass
class MapLayout:
    obstacles: list                      # Obstacle (water, stone)
    shelters: list                       # Obstacle (litter)
    trees: list                          # (x, y)
    nest: tuple
    spider_start: tuple
    centipede_start: tuple
    biomes: list = field(default_factory=list)     # {"kind", "x", "y", "r"}
    name: str = ""
    flowers: list | None = None          # initial flower spots; None: the arena picks free spots
    den: tuple | None = None             # centipede den: newcomers may come out of it
    moisture: np.ndarray | None = None   # fields on the GRID (rows = y), for checks and pictures
    fertility: np.ndarray | None = None


# --- noise --------------------------------------------------------------------------
def value_noise(rng, nx: int, ny: int, octaves=((120.0, 1.0), (60.0, 0.5), (30.0, 0.25))) -> np.ndarray:
    """Smooth noise 0..1 on an (ny, nx) grid of GRID mm cells: random lattice values, smoothstep interpolation."""
    xs, ys = (np.arange(nx) + 0.5) * GRID, (np.arange(ny) + 0.5) * GRID
    out = np.zeros((ny, nx))
    for scale, amp in octaves:
        lx, ly = int(np.ceil(nx * GRID / scale)) + 2, int(np.ceil(ny * GRID / scale)) + 2
        lat = rng.random((ly, lx))
        ox, oy = rng.uniform(0, 1, 2)                       # random lattice offset
        u, v = xs / scale + ox, ys / scale + oy
        i, j = u.astype(int), v.astype(int)
        fu, fv = u - i, v - j
        fu, fv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)
        a = lat[j[:, None], i[None]] * (1 - fu[None]) + lat[j[:, None], i[None] + 1] * fu[None]
        b = lat[j[:, None] + 1, i[None]] * (1 - fu[None]) + lat[j[:, None] + 1, i[None] + 1] * fu[None]
        out += amp * (a * (1 - fv[:, None]) + b * fv[:, None])
    out -= out.min()
    return out / max(out.max(), 1e-9)


def sample(fieldg: np.ndarray, x, y):
    """Field value at points (nearest grid cell)."""
    ny, nx = fieldg.shape
    i = np.clip((np.asarray(x) / GRID).astype(int), 0, nx - 1)
    j = np.clip((np.asarray(y) / GRID).astype(int), 0, ny - 1)
    return fieldg[j, i]


def bump(nx, ny, x, y, r):
    xs, ys = (np.arange(nx) + 0.5) * GRID, (np.arange(ny) + 0.5) * GRID
    return np.exp(-((xs[None] - x) ** 2 + (ys[:, None] - y) ** 2) / (2 * r * r))


# --- geometry -----------------------------------------------------------------------
def _grid_blocked(obstacles, nx, ny, pad=2.0) -> np.ndarray:
    xs, ys = (np.arange(nx) + 0.5) * GRID, (np.arange(ny) + 0.5) * GRID
    X, Y = np.meshgrid(xs, ys)
    out = np.zeros((ny, nx), dtype=bool)
    for o in obstacles:
        out |= o.inside(X, Y, pad)
    return out


def connected(free: np.ndarray) -> bool:
    """All free cells form one 4-connected component."""
    if not free.any():
        return False
    seed = np.unravel_index(np.argmax(free), free.shape)
    reach = np.zeros_like(free)
    reach[seed] = True
    while True:
        grown = reach.copy()
        grown[1:] |= reach[:-1]
        grown[:-1] |= reach[1:]
        grown[:, 1:] |= reach[:, :-1]
        grown[:, :-1] |= reach[:, 1:]
        grown &= free
        if (grown == reach).all():
            break
        reach = grown
    return bool((reach == free).all())


def walk_grid(obstacles, W, H, pad=2.0):
    nx, ny = int(np.ceil(W / GRID)), int(np.ceil(H / GRID))
    free = ~_grid_blocked(obstacles, nx, ny, pad)
    free[0], free[-1], free[:, 0], free[:, -1] = False, False, False, False    # the map edge
    return free


def _clearance(o, others) -> float:
    """Free distance between bounding circles (negative = overlap)."""
    if not others:
        return np.inf
    d = [np.hypot(o.x - p.x, o.y - p.y) - max(o.rx, o.ry) - max(p.rx, p.ry) for p in others]
    return float(min(d))


def _best(rng, n, W, H, margin, score):
    """n candidate points sorted by score (+ a little noise), best first."""
    x = rng.uniform(margin, W - margin, n)
    y = rng.uniform(margin, H - margin, n)
    s = score(x, y) + 0.08 * rng.random(n)
    k = np.argsort(-s)
    return x[k], y[k]


# --- generation ---------------------------------------------------------------------
def generate(cfg, rng: np.random.Generator, biome: str | None = None) -> MapLayout:
    from .arena import Obstacle     # late import: arena imports this module
    W, H = cfg.width, cfg.height
    nx, ny = int(np.ceil(W / GRID)), int(np.ceil(H / GRID))
    biome = biome or getattr(cfg, "biome", None) or str(rng.choice(BIOMES))
    if biome not in BIOMES:
        raise ValueError(f"unknown biome {biome!r}, expected one of {BIOMES}")
    moist, fert = value_noise(rng, nx, ny), value_noise(rng, nx, ny)
    # the special biome region: where its field already leans the right way
    br = float(rng.uniform(85, 115))
    want = {"orchard": fert, "meadow": fert * (1 - 0.5 * np.abs(moist - 0.5)), "marsh": moist, "rocky": 1 - moist}[biome]
    cx, cy = _best(rng, 60, W, H, br * 0.8, lambda x, y: sample(want, x, y))
    bx, by = float(cx[0]), float(cy[0])
    b = bump(nx, ny, bx, by, br * 0.7)
    if biome == "marsh":
        moist = np.clip(moist * 0.5 + 0.8 * b, 0, 1)
    elif biome == "rocky":
        moist = np.clip(moist * (1 - 0.8 * b), 0, 1)
        fert = np.clip(fert * (1 - 0.6 * b), 0, 1)
    else:
        fert = np.clip(fert * 0.7 + 0.5 * b, 0, 1)
    lay = None
    for attempt in range(40):                               # re-roll; late attempts are sparse (fewer, smaller ponds)
        lay = _try_layout(cfg, rng, biome, moist, fert, (bx, by, br), Obstacle, sparse=attempt >= 30)
        if lay is not None:
            break
    if lay is None:                                         # never seen in tests; keep the world startable
        lay = classic(cfg)
        lay.flowers = None
    lay.moisture, lay.fertility = moist, fert
    lay.biomes = [{"kind": biome, "x": round(bx, 1), "y": round(by, 1), "r": round(br, 1)}]
    lay.name = BIOME_RU[biome]
    return lay


def _try_layout(cfg, rng, biome, moist, fert, region, Obstacle, sparse=False):
    W, H, crown = cfg.width, cfg.height, cfg.tree_crown
    bx, by, br = region
    in_region = lambda x, y: np.hypot(x - bx, y - by) < br   # noqa: E731
    obstacles: list = []

    def ok_connect(extra):
        return connected(walk_grid(obstacles + [extra], W, H, pad=3.0))

    # ponds: wettest ground first
    if biome == "marsh":        # one medium pond and many small ones, mostly inside the marsh
        sizes = [rng.uniform(34, 45)] + sorted(rng.uniform(16, 30, int(rng.integers(4, 8))), reverse=True)
    elif biome == "rocky":
        sizes = [rng.uniform(34, 45)]
    else:                       # 1-3 ponds like the classic map (radii 34-72 mm)
        sizes = sorted(list(rng.uniform(46, 72, int(rng.integers(1, 3)))) + list(rng.uniform(34, 45, int(rng.integers(0, 2)))),
                       reverse=True)
    if sparse:
        sizes = sizes[:1] if biome != "marsh" else sizes[:4]
    pond_score = (lambda x, y: sample(moist, x, y) + 0.4 * in_region(x, y)) if biome == "marsh" else \
        (lambda x, y: sample(moist, x, y))
    for r in sizes:
        placed = False
        for shrink in (1.0, 0.85, 0.7):
            rx = r * shrink
            ry = rx / rng.uniform(1.1, 1.2)
            cx, cy = _best(rng, 300, W, H, EDGE + rx, pond_score)
            for x, y in zip(cx[:80], cy[:80]):
                o = Obstacle("water", float(x), float(y), float(rx), float(ry), float(rng.uniform(-0.4, 0.4)))
                if _clearance(o, obstacles) < (GAP if biome != "marsh" else 10.0):
                    continue
                if ok_connect(o):
                    obstacles.append(o)
                    placed = True
                    break
            if placed:
                break
    ponds = [o for o in obstacles if o.kind == "water"]
    if not ponds:
        return None
    water_area = sum(np.pi * o.rx * o.ry for o in ponds)
    if water_area > 0.16 * W * H:
        return None

    def crown_wet(x, y):
        a = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        px = np.concatenate([[x], x + crown * 0.5 * np.cos(a), x + crown * np.cos(a)])
        py = np.concatenate([[y], y + crown * 0.5 * np.sin(a), y + crown * np.sin(a)])
        hit = np.zeros(len(px), dtype=bool)
        for o in obstacles:
            hit |= o.inside(px, py, 2.0)
        out = (px < 8) | (px > W - 8) | (py < 8) | (py > H - 8)
        return (hit | out).mean()

    # apple trees: fertile ground, crowns mostly clear of water
    trees: list = []
    n_trees = int(rng.integers(3, 5)) if biome == "orchard" else int(rng.integers(1, 4))
    if biome == "orchard":
        for _ in range(n_trees):
            for _ in range(60):
                a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0, br * 0.75)
                x, y = bx + d * np.cos(a), by + d * np.sin(a)
                if not (crown * 0.6 < x < W - crown * 0.6 and crown * 0.6 < y < H - crown * 0.6):
                    continue
                if any(np.hypot(x - tx, y - ty) < crown * 1.2 for tx, ty in trees) or crown_wet(x, y) > 0.1:
                    continue
                trees.append((float(x), float(y)))
                break
        if len(trees) < (2 if sparse else 3):
            return None
        if rng.random() < 0.5:
            n_trees = len(trees) + 1                        # sometimes one more tree elsewhere
    cx, cy = _best(rng, 400, W, H, crown * 0.6, lambda x, y: sample(fert, x, y))
    for x, y in zip(cx, cy):
        if len(trees) >= n_trees:
            break
        if any(np.hypot(x - tx, y - ty) < crown * 2.2 for tx, ty in trees) or crown_wet(x, y) > 0.08:
            continue
        if biome == "rocky" and in_region(x, y):
            continue
        trees.append((float(x), float(y)))
    if not trees:
        return None

    # stones: scattered, out of tree crowns (a few may touch the rim)
    if biome == "rocky":
        n_st = int(rng.integers(10, 17))
    elif biome == "marsh":
        n_st = int(rng.integers(1, 4))
    else:
        n_st = int(rng.integers(4, 9))
    for k in range(n_st):
        r = float(rng.uniform(10, 20))
        for _ in range(60):
            if biome == "rocky" and k < n_st * 0.6:        # most of them in the rocky patch
                a, d = rng.uniform(0, 2 * np.pi), br * np.sqrt(rng.random())
                x, y = bx + d * np.cos(a), by + d * np.sin(a)
            else:
                x, y = rng.uniform(15 + r, W - 15 - r), rng.uniform(15 + r, H - 15 - r)
            if not (15 + r < x < W - 15 - r and 15 + r < y < H - 15 - r):
                continue
            o = Obstacle("stone", float(x), float(y), r, r, 0.0)
            if _clearance(o, obstacles) < GAP or any(np.hypot(x - tx, y - ty) < crown * 0.8 + r for tx, ty in trees):
                continue
            if ok_connect(o):
                obstacles.append(o)
                break
    n_stones = sum(o.kind == "stone" for o in obstacles)
    if n_stones < {"rocky": 8, "marsh": 1}.get(biome, 4) * (0.5 if sparse else 1):
        return None

    # ant nest: dry ground far from water and trees
    def dist_water(x, y):
        return min(np.hypot(x - o.x, y - o.y) - max(o.rx, o.ry) for o in ponds)

    nest = None
    cx, cy = _best(rng, 300, W, H, 30, lambda x, y: 1 - sample(moist, x, y))
    for x, y in zip(cx, cy):
        if dist_water(x, y) < 60 or any(np.hypot(x - o.x, y - o.y) < max(o.rx, o.ry) + 22 for o in obstacles):
            continue
        if any(np.hypot(x - tx, y - ty) < crown + 15 for tx, ty in trees):
            continue
        nest = (float(x), float(y))
        break
    if nest is None:
        return None

    # leaf litter: moist ground near trees
    shelters: list = []
    tx_, ty_ = np.array([t[0] for t in trees]), np.array([t[1] for t in trees])

    def litter_score(x, y):
        d = np.hypot(np.asarray(x)[..., None] - tx_, np.asarray(y)[..., None] - ty_).min(-1)
        return 0.6 * sample(moist, x, y) + np.exp(-d / 70.0)

    n_lit = int(rng.integers(4, 8))
    cx, cy = _best(rng, 400, W, H, 30, litter_score)
    for x, y in zip(cx, cy):
        if len(shelters) >= n_lit:
            break
        rx, ry = float(rng.uniform(18, 34)), float(rng.uniform(14, 30))
        o = Obstacle("litter", float(x), float(y), rx, ry, float(rng.uniform(-0.5, 0.5)))
        if _clearance(o, obstacles) < 3 or _clearance(o, shelters) < 10 or np.hypot(x - nest[0], y - nest[1]) < max(rx, ry) + 25:
            continue
        if not (x - max(rx, ry) > 5 and x + max(rx, ry) < W - 5 and y - max(rx, ry) > 5 and y + max(rx, ry) < H - 5):
            continue
        shelters.append(o)

    def free_at(x, y, pad):
        return all(not o.inside(x, y, pad) for o in obstacles) and 12 < x < W - 12 and 12 < y < H - 12

    # predators: spider (with its first web) and the first centipede
    web_pad = max(cfg.web_radius) * 0.6 + 2
    spider = den = None
    if biome == "orchard":
        tries = [(bx + d * np.cos(a), by + d * np.sin(a)) for a, d in zip(rng.uniform(0, 2 * np.pi, 80), rng.uniform(0, br, 80))]
    else:
        cx, cy = _best(rng, 200, W, H, 40, lambda x, y: 0.5 * sample(fert, x, y))
        tries = list(zip(cx, cy))
    for x, y in tries:
        if free_at(x, y, web_pad) and np.hypot(x - nest[0], y - nest[1]) > 70 and \
                min(np.hypot(x - tx, y - ty) for tx, ty in trees) > 20:
            spider = (float(x), float(y))
            break
    if spider is None:
        return None
    if biome == "orchard":
        tries = [(bx + d * np.cos(a), by + d * np.sin(a)) for a, d in
                 zip(rng.uniform(0, 2 * np.pi, 80), rng.uniform(br * 0.6, br * 1.3, 80))]
    else:
        tries = list(zip(rng.uniform(40, W - 40, 200), rng.uniform(40, H - 40, 200)))
    centipede = None
    for x, y in tries:
        if free_at(x, y, 8) and np.hypot(x - spider[0], y - spider[1]) > (50 if biome == "orchard" else 120) and \
                np.hypot(x - nest[0], y - nest[1]) > 60:
            centipede = (float(x), float(y))
            break
    if centipede is None:
        return None
    if biome == "orchard":
        den = centipede

    # the map must stay walkable with room to spare
    free = walk_grid(obstacles, W, H, pad=3.0)
    if not connected(free) or free.mean() < 0.72:
        return None

    # initial flowers: fertile ground (meadow and orchard get more)
    n_fl = cfg.initial_flowers + {"meadow": 6, "orchard": 3}.get(biome, 0)
    n_fl = min(n_fl, cfg.max_flowers)
    flowers: list = []
    px, py = rng.uniform(20, W - 20, 3000), rng.uniform(20, H - 20, 3000)
    wgt = sample(fert, px, py) ** 2 + 0.02
    wgt *= np.where(in_region(px, py), {"meadow": 4.0, "orchard": 2.5, "rocky": 0.2}.get(biome, 1.0), 1.0)
    pick = rng.choice(len(px), size=40 * n_fl, replace=False, p=wgt / wgt.sum())
    for x, y in zip(px[pick], py[pick]):
        if len(flowers) >= n_fl:
            break
        if not free_at(x, y, 8) or np.hypot(x - nest[0], y - nest[1]) < 25 or \
                any(np.hypot(x - fx, y - fy) < 12 for fx, fy in flowers):
            continue
        flowers.append((float(x), float(y)))
    return MapLayout(obstacles, shelters, trees, nest, spider, centipede, flowers=flowers, den=den)


def classic(cfg) -> MapLayout:
    """The old fixed layout (fractions of width/height)."""
    from .arena import Obstacle
    W, H = cfg.width, cfg.height
    # proportions follow the viewer sprites (lake ~1.15:1, stones roughly round)
    obstacles = [
        Obstacle("water", W * 0.52, H * 0.55, 72, 60, 0.0),
        Obstacle("water", W * 0.17, H * 0.24, 40, 34, 0.0),
        Obstacle("stone", W * 0.35, H * 0.74, 14, 14, 0.0),
        Obstacle("stone", W * 0.80, H * 0.27, 19, 19, 0.0),
        Obstacle("stone", W * 0.84, H * 0.80, 12, 12, 0.0),
        Obstacle("stone", W * 0.10, H * 0.72, 16, 16, 0.0),
        Obstacle("stone", W * 0.64, H * 0.14, 11, 11, 0.0),
    ]
    shelters = [
        Obstacle("litter", W * 0.30, H * 0.80, 30, 18, 0.3),
        Obstacle("litter", W * 0.06, H * 0.55, 22, 34, 0.0),
        Obstacle("litter", W * 0.72, H * 0.88, 34, 16, -0.2),
        Obstacle("litter", W * 0.88, H * 0.40, 20, 28, 0.4),
        Obstacle("litter", W * 0.45, H * 0.12, 30, 14, 0.1),
    ]
    return MapLayout(obstacles, shelters, [(W * 0.70, H * 0.30), (W * 0.22, H * 0.56)], (W * 0.93, H * 0.08),
                     (W * 0.30, H * 0.42), (W * 0.85, H * 0.55), biomes=[], name=BIOME_RU["classic"])


def validate(cfg, lay: MapLayout) -> list:
    """Problems of a layout (empty list = valid)."""
    W, H = cfg.width, cfg.height
    bad = []
    free = walk_grid(lay.obstacles, W, H, pad=3.0)
    if not connected(free):
        bad.append("free ground not connected")
    if free.mean() < 0.7:
        bad.append(f"free fraction {free.mean():.2f}")
    obs = lay.obstacles
    for i, o in enumerate(obs):
        if _clearance(o, obs[:i] + obs[i + 1:]) < 0:
            bad.append(f"overlap {o.kind} at {o.x:.0f},{o.y:.0f}")
        if o.kind == "water" and min(o.x - o.rx, o.y - o.rx, W - o.x - o.rx, H - o.y - o.rx) < EDGE - 1e-6:
            bad.append("pond too close to the edge")
    ponds = [o for o in obs if o.kind == "water"]
    if not ponds:
        bad.append("no pond")
    for x, y, what in [(*lay.nest, "nest"), (*lay.spider_start, "spider"), (*lay.centipede_start, "centipede")] + \
            [(tx, ty, "tree") for tx, ty in lay.trees] + [(fx, fy, "flower") for fx, fy in (lay.flowers or [])]:
        if any(o.inside(x, y, 1.0) for o in obs):
            bad.append(f"{what} inside an obstacle")
    return bad
