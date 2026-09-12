"""Seeded maze world for the Eye + Nose flies.

The maze is a tile grid: 1 = wall, 0 = floor. Maze cells sit at odd
coordinates, the tiles between them are either wall or open corridor.

Rules:
- food smells through walls (intensity falls with straight-line distance);
- food and traps are seen only along a clear straight corridor;
- stepping on a trap kills the Nose fly, reaching food wins.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# Headings: 0 = north, 1 = east, 2 = south, 3 = west; offsets as (row, col).
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
FORWARD, LEFT, RIGHT = 0, 1, 2


@dataclass
class MazeConfig:
    width: int = 8             # maze cells horizontally
    height: int = 8            # maze cells vertically
    loops: float = 0.2         # fraction of inner walls removed after carving
    n_traps: int = 4
    min_food_dist: int = 10    # minimal path length (tiles) from start to food
    vision_range: int = 6      # tiles
    smell_scale: float = 6.0   # tiles; smell = exp(-dist / smell_scale)


@dataclass
class Map:
    seed: int
    cfg: MazeConfig
    grid: np.ndarray                      # (rows, cols), 1 = wall
    start: tuple[int, int]
    heading: int
    food: tuple[int, int]
    traps: set[tuple[int, int]] = field(default_factory=set)

    def is_wall(self, pos) -> bool:
        r, c = pos
        return not (0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]) or self.grid[r, c] == 1


def generate(seed: int, cfg: MazeConfig | None = None) -> Map:
    cfg = cfg or MazeConfig()
    rng = np.random.default_rng(seed)
    grid = _carve(cfg.width, cfg.height, rng)
    _add_loops(grid, cfg.loops, rng)

    cells = [(r, c) for r in range(1, grid.shape[0], 2) for c in range(1, grid.shape[1], 2)]
    start = cells[rng.integers(len(cells))]
    dist = bfs(grid, start)
    far = [p for p in cells if dist.get(p, -1) >= cfg.min_food_dist]
    if not far:  # tiny maze: take the farthest cell
        far = [max(cells, key=lambda p: dist[p])]
    food = far[rng.integers(len(far))]

    m = Map(seed, cfg, grid, start, int(rng.integers(4)), food)
    _place_traps(m, rng)
    return m


def _carve(w: int, h: int, rng) -> np.ndarray:
    """Recursive backtracker: a perfect maze, every cell reachable."""
    grid = np.ones((2 * h + 1, 2 * w + 1), dtype=np.int8)
    stack = [(1, 1)]
    grid[1, 1] = 0
    while stack:
        r, c = stack[-1]
        nbrs = [(r + 2 * dr, c + 2 * dc) for dr, dc in DIRS
                if 0 < r + 2 * dr < grid.shape[0] and 0 < c + 2 * dc < grid.shape[1]
                and grid[r + 2 * dr, c + 2 * dc] == 1]
        if not nbrs:
            stack.pop()
            continue
        nr, nc = nbrs[rng.integers(len(nbrs))]
        grid[(r + nr) // 2, (c + nc) // 2] = 0
        grid[nr, nc] = 0
        stack.append((nr, nc))
    return grid


def _add_loops(grid: np.ndarray, fraction: float, rng):
    """Remove some inner walls between cells, so the maze has several routes."""
    rows, cols = grid.shape
    walls = [(r, c) for r in range(1, rows - 1) for c in range(1, cols - 1)
             if grid[r, c] == 1 and (r % 2) != (c % 2)]
    for i in rng.permutation(len(walls))[: int(len(walls) * fraction)]:
        r, c = walls[i]
        grid[r, c] = 0


def _place_traps(m: Map, rng):
    """Add traps one by one, keeping a trap-free path from start to food.

    Traps far from the route would never matter, so tiles on and near the
    shortest path to food are tried first (with some randomness).
    """
    route = shortest_path(m.grid, m.start, m.food)
    to_route = bfs(m.grid, route)
    near_start = {(m.start[0] + dr, m.start[1] + dc) for dr, dc in DIRS} | {m.start}
    candidates = [p for p in to_route if p not in near_start and p != m.food]
    candidates.sort(key=lambda p: to_route[p] + 3 * rng.random())
    for trap in candidates:
        if len(m.traps) >= m.cfg.n_traps:
            break
        if m.food in bfs(m.grid, m.start, blocked=m.traps | {trap}):
            m.traps.add(trap)


def shortest_path(grid: np.ndarray, start, goal) -> list:
    prev = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        for dr, dc in DIRS:
            p = (cur[0] + dr, cur[1] + dc)
            if p not in prev and grid[p] == 0:
                prev[p] = cur
                queue.append(p)
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def bfs(grid: np.ndarray, start, blocked=frozenset()) -> dict:
    """Path distance (tiles) from start (a tile or a list of tiles) to every reachable floor tile."""
    sources = start if isinstance(start, list) else [start]
    dist = {s: 0 for s in sources}
    queue = deque(sources)
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS:
            p = (r + dr, c + dc)
            if p not in dist and p not in blocked and grid[p] == 0:
                dist[p] = dist[(r, c)] + 1
                queue.append(p)
    return dist


# --- Senses -----------------------------------------------------------------

def smell(m: Map, pos, heading: int) -> tuple[float, float]:
    """Food odor at the left and right antenna. Walls do not block it."""
    fr, fc = DIRS[heading]
    lr, lc = DIRS[(heading - 1) % 4]
    out = []
    for side in (1, -1):  # left, right
        # antennae sit slightly forward and to each side of the body
        r = pos[0] + 0.3 * fr + side * 0.3 * lr
        c = pos[1] + 0.3 * fc + side * 0.3 * lc
        d = np.hypot(r - m.food[0], c - m.food[1])
        out.append(float(np.exp(-d / m.cfg.smell_scale)))
    return out[0], out[1]


@dataclass
class Ray:
    wall_dist: int         # tiles until the wall
    food_dist: int | None  # tiles until food, if visible
    trap_dist: int | None  # tiles until the nearest trap, if visible


def vision(m: Map, pos, heading: int) -> list[Ray]:
    """Look along the corridor: forward, left, right, back (relative to heading)."""
    rays = []
    for turn in (0, -1, 1, 2):
        dr, dc = DIRS[(heading + turn) % 4]
        food = trap = None
        d = 0
        r, c = pos
        while d < m.cfg.vision_range:
            r, c = r + dr, c + dc
            if m.is_wall((r, c)):
                break
            d += 1
            if food is None and (r, c) == m.food:
                food = d
            if trap is None and (r, c) in m.traps:
                trap = d
        rays.append(Ray(d, food, trap))
    return rays


# --- Movement -----------------------------------------------------------------

def step(m: Map, pos, heading: int, action: int):
    """Apply an action. Returns (pos, heading, event), event in
    {"move", "turn", "bump", "food", "trap"}."""
    if action == LEFT:
        return pos, (heading - 1) % 4, "turn"
    if action == RIGHT:
        return pos, (heading + 1) % 4, "turn"
    dr, dc = DIRS[heading]
    nxt = (pos[0] + dr, pos[1] + dc)
    if m.is_wall(nxt):
        return pos, heading, "bump"
    if nxt == m.food:
        return nxt, heading, "food"
    if nxt in m.traps:
        return nxt, heading, "trap"
    return nxt, heading, "move"


# --- Rendering ----------------------------------------------------------------

def render(m: Map, pos=None, heading=None) -> str:
    pos = pos or m.start
    heading = m.heading if heading is None else heading
    arrows = "▲▶▼◀"
    lines = []
    for r in range(m.grid.shape[0]):
        row = []
        for c in range(m.grid.shape[1]):
            p = (r, c)
            if p == pos:
                row.append(arrows[heading] + " ")
            elif p == m.food:
                row.append("🍯")
            elif p in m.traps:
                row.append("💀")
            elif m.grid[r, c]:
                row.append("██")
            else:
                row.append("  ")
        lines.append("".join(row))
    return "\n".join(lines)
