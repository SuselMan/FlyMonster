"""Flies with whole-brain simulations living in the forest-floor world.

Loop every world step (10 ms): world -> sensory rates -> 20 brain steps
(dt 0.5 ms) -> descending-neuron spike counts -> low-pass readout -> body.

What is connectome and what is ours (see README):
- Sensory transduction, the body, physiology (energy, eggs) and the other
  animals are our models. Between sensory neurons and descending/motor
  neurons it is the connectome.
- Body mapping uses descending neurons with published roles only:
  DNa01/DNa02 ipsilateral turning while walking, MDN backward walking,
  DNp01 giant fiber short escape hop, DNp02/DNp04/DNp11 long-mode takeoff
  (a flight whose duration the body sets; in the air the same steering
  command is spent in body saccades), MN9
  proboscis (feeding). DNp09 is shown but drives nothing: its effect depends
  on context. Walking itself is an innate generator (leg circuits are in the
  ventral nerve cord, absent from FAFB).
- Web: sticky. A stuck fly struggles on its own (body) and takeoff attempts of
  the brain add escape force; enough force tears the fly free (an old, weakened
  web needs less, and tearing damages it), otherwise the spider arrives after a
  reaction delay. No dice roll.
- Water: humidity around ponds (arena) reaches the sacculus moist cells
  (HRN_VP5, TRN_VP1m) through the same saturating transducer as odors; thirst
  raises the gain. Whether the brain then steers towards water is the
  connectome's business (scripts/hygro_scan.py).
- Food is finite: flies (and ants) eat it away. Vision sees flies, food,
  stones, the predators and ants as objects; there is no food-seeking rule.
- Pollen is body state: a fly that fed on a flower carries its pollen and
  pollinates the next different flower it feeds on. Nothing steers it there.
- Flight metrics (long flights, flying across water) are only counted, they
  do not change how flights are triggered.
- Hunger lowers the threshold of NPF neurons and raises sugar sensitivity.
- Taste and grooming use the neuron sets of Shiu et al. 2024: bitter GRNs on
  rotting food, water GRNs at a pond edge (thirst raises their rate), Johnston's
  organ JON-CE driven by dust on the antennae; the body drinks with MN9 and grooms
  (stops, loses dust and pollen) with aBN1. Which food is bitter, how dust
  accumulates and the thirst gain are ours.
- Reproduction is clonal; a mature egg is laid while the fly feeds.
- Cold (body model, ours): below ~18 C a fly walks slower, below chill_temp it falls into chill
  coma (no walking, takeoff or feeding, low metabolism) until it warms up. Leaf litter and stones
  are warmer in the cold (arena microclimate). The brain does not sense temperature yet.
- Mushroom-body learning (LifeConfig.learning, needs the MB physiology): dopamine-gated
  depression of Kenyon cell -> MBON synapses, one set of multipliers per fly
  (flysim/plasticity.py). The world supplies what the connectome does not
  (scripts/dan_drive_scan.py): sugar on the labellum drives the reward DANs
  (PAM), bitter taste, a web and a diving predator drive the punishment DANs
  (PPL1). A newborn fly starts with no memories.
- The genome holds physiological parameters only.
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from .. import connectome, neurons
from ..body import dn_reach
from ..brain import FlyBrain
from ..fastbrain import FastBrain
from ..physiology import DEFAULT, Physiology, apply, neuron_meta
from ..plasticity import MushroomBody
from ..senses import HUMIDITY, Compass, Olfaction, Wind
from .arena import Arena, ArenaConfig

WORLD_DT = 0.010          # s
READ_TAU = 0.050          # s, readout low-pass
READOUT = {               # name -> (cell_type, side or None)
    "DNa01 L": ("DNa01", "left"), "DNa01 R": ("DNa01", "right"),
    "DNa02 L": ("DNa02", "left"), "DNa02 R": ("DNa02", "right"),
    "MDN": ("MDN", None), "GF": ("DNp01", None), "MN9": (None, None), "aBN1": (None, None),
    "DNp02": ("DNp02", None), "DNp04": ("DNp04", None), "DNp11": ("DNp11", None), "DNp09": ("DNp09", None),
}
# Steering: the ventral nerve cord that turns DN activity into leg movements is not in FAFB, so
# the mapping is evolved (scripts/evolve_steer.py): turn = sum_k w_k * (L - R rate of DN type k,
# high-passed with turn_adapt) / 50 + bias. The brain itself is untouched.
STEER_TYPES = ("DNa01", "DNa02", "DNb05", "DNp18", "DNg99", "DNb06", "DNbe001", "DNp33", "DNg13", "DNp35",
               "DNg56", "DNp06", "DNp31", "DNa04", "DNa10", "DNp19", "DNg96", "DNg31", "DNp73", "DNa11")
for _t in STEER_TYPES:
    READOUT.setdefault(f"{_t} L", (_t, "left"))
    READOUT.setdefault(f"{_t} R", (_t, "right"))
STEER_EVOLVED = {"DNa02": 2.09, "DNg99": 2.55}   # elite mean of both islands; validation (72 episodes each):
# reached the apple 54% vs 8% pure wander and 0% legacy DNa01+DNa02 (results/eval_seed12.json)
LONG_FLIGHT_MM = 80.0     # a long-mode flight that covered at least this much ground
GENES = ("vision", "smell", "taste", "looming", "hunger", "walk")
SENSES = ("fruit L", "fruit R", "vinegar L", "vinegar R", "predator", "loom", "touch", "sugar", "bitter", "water", "dust",
          "humid")


@dataclass
class LifeConfig:
    n_flies: int = 24
    max_flies: int = 32
    seed: int = 1
    walk_speed: float = 14.0      # mm/s, innate walking generator
    turn_gain: float = 0.025      # rad/s per Hz of left-right DNa01+DNa02 difference
    turn_adapt: float = 5.0       # s, adaptation of the body to a sustained left-right difference
    steer: dict | None = field(default_factory=lambda: dict(STEER_EVOLVED))   # None -> legacy DNa01+DNa02 * turn_gain
    wander: float = 1.2           # rad/s^0.5, turning noise of the walking generator
    gf_threshold: float = 60.0    # Hz, giant fiber -> short escape hop
    long_threshold: float = 45.0  # Hz, mean of DNp02/04/11 -> long-mode takeoff
    mdn_threshold: float = 25.0   # Hz, backward walking
    feed_threshold: float = 15.0  # Hz, MN9 while touching food or water
    groom_threshold: float = 8.0  # Hz, aBN1 -> start grooming
    chill_temp: float = 7.0       # deg C, chill coma below (1 deg hysteresis)
    warm_temp: float = 18.0       # deg C, full walking speed above
    coma_metabolism: float = 0.25 # fraction of energy use in chill coma
    thirst_s: float = 1500.0      # s, full -> dry
    drink_per_s: float = 0.03     # hydration per s while drinking
    juice: float = 0.004          # hydration per sugar unit eaten (fruit, nectar)
    dust_ground: float = 0.002    # dust per s while walking
    dust_litter: float = 0.015    # dust per s while walking in leaf litter
    dust_flower: float = 0.05     # dust (pollen) per s while feeding on a flower
    groom_clean: float = 0.25     # dust removed per s of grooming
    hop: tuple = (0.25, 100.0)    # s, mm/s of a giant-fiber hop
    flight: tuple = (1.6, 70.0)   # s, mm/s of a long-mode flight
    saccade_rate: float = 1.0     # per s in flight: spontaneous body saccades on top of the steered ones
    saccade_angle: tuple = (0.5, 1.6)   # rad, size of a spontaneous saccade (steered ones are capped at the max)
    air_steer: float = 1.0        # gain of the brain's turn command in flight
    saccade_threshold: float = 0.6  # rad of accumulated turn intent that fires a steered saccade
    web_escape: float = 1.3       # escape force needed to tear free (fresh web: 2-3 takeoff attempts)
    struggle: float = 0.10        # escape force per s a stuck fly gains by struggling (fresh web alone: ~19 s)
    escape_decay: float = 0.03    # escape force lost per s
    humid_gain: tuple = (0.3, 0.7)  # humidity drive to the moist cells: base + thirst part
    metabolism: float = 0.0005
    move_cost: float = 0.00004
    jump_cost: float = 0.01
    flight_cost: float = 0.03
    sugar_per_s: float = 5.0
    energy_per_sugar: float = 0.01
    lifespan: tuple = (2700.0, 3600.0)
    maturity: float = 300.0
    egg_interval: float = 120.0
    egg_energy: float = 0.12
    egg_min_energy: float = 0.3
    max_eggs: int = 12            # eggs wait for a free brain slot (the oldest is dropped beyond this)
    hatch_min_temp: float = 8.0   # deg C at the egg: no hatching in the cold
    hatch_time: float = 90.0
    mutation: float = 0.12
    physiology: Physiology = DEFAULT
    learning: bool = False        # mushroom-body plasticity (with the MB physiology)
    dan_hz: float = 100.0         # PAM / PPL1 drive at full reward / punishment
    eta: float = 0.0005           # learning rate (scripts/mb_learning_check.py: odor-specific after 3 pairings)
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
        # Brains live in a fixed batch of max_flies slots; flies take and free slots.
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        params = cfg.physiology.lif()
        self.steps_per_world = round(WORLD_DT * 1000 / params.dt)
        self._wire()
        # sensory transducers run on the CPU (inputs come from numpy); rates reach the GPU in one pinned,
        # non-blocking copy per step, so the CPU never waits for the brain block that is still running
        cpu = torch.device("cpu")
        self.olf = Olfaction(self.meta, cfg.max_flies, cpu)
        self.hyg = Olfaction(self.meta, cfg.max_flies, cpu, odorants=HUMIDITY, tau_adapt_ms=6000.0, gamma=0.5)
        self.wind_sense = Wind(self.meta, cpu)
        self.compass = Compass(self.con, self.meta, cpu)
        self.sense_col_cpu = self.sense_col.cpu()
        self.input_idx = torch.cat([self.sense_idx.cpu(), self.olf.input_idx, self.hyg.input_idx,
                                    self.wind_sense.input_idx, self.compass.input_idx]).to(self.dev)
        self.read_mix_cpu = self.read_mix.cpu()
        self.n_read = len(self.read_idx)
        self.mb = None
        if cfg.learning:
            if not cfg.physiology.kc_cholinergic:
                raise ValueError("learning needs a physiology with living Kenyon cells (physiology.MB)")
            self.mb = MushroomBody(model, self.meta, self.dev, raw=self.con, eta=cfg.eta)
            # the plasticity reads Kenyon cell and dopamine neuron spikes after the body readouts
            self.read_idx = torch.cat([self.read_idx, torch.tensor(self.mb.kc_idx, device=self.dev),
                                       torch.tensor(self.mb.dan_idx, device=self.dev)])
        pin = self.dev.type == "cuda"
        # two alternating pinned buffers: a non-blocking copy may still be queued when the next step writes
        self._rates_host = [torch.zeros(len(self.input_idx), cfg.max_flies, pin_memory=pin) for _ in range(2)]
        self._bias_host = [torch.zeros(cfg.max_flies, pin_memory=pin) for _ in range(2)]
        self._host_turn = 0
        if self.dev.type == "cuda" and params.std_u == 0:
            self.brain = FastBrain(model, cfg.max_flies, params, self.input_idx, self.read_idx, self.npf_idx,
                                   steps=self.steps_per_world, max_spikes=96 * cfg.max_flies, max_events=20_000 * cfg.max_flies)
            if self.mb is not None:
                self.mb.attach(self.brain)
        else:
            self.brain = FlyBrain(model, batch=cfg.max_flies, params=params, device=self.dev)
            if self.mb is not None:
                self.mb.attach(self.brain)
        self.free_slots = list(range(cfg.max_flies))
        self._pending = None
        self.arena = Arena.create(cfg.arena, cfg.seed)
        self.t = 0.0
        self.ids, self.next_id = [], 0
        self.eggs: list[Egg] = []
        self.events: list = []
        self.dead: list = []
        self.births = 0
        # stuck: id+1 of the web a fly is stuck on (0 = free)
        self.counters = {"hops": 0, "flights": 0, "long_flights": 0, "water_crossings": 0, "landings_by_food": 0, "eaten": 0.0}
        self._columns = ("x", "y", "heading", "energy", "age", "lifespan", "air_left", "air_total", "air_speed",
                         "air_height", "cooldown", "jumped_at", "state", "turn_base", "turn_intent", "last_feed", "last_egg",
                         "generation", "parent", "stuck", "escape_force", "touch_left", "slot", "genome", "senses",
                         "air_x0", "air_y0", "air_water", "air_long", "pollen", "pollen_t",
                         "meals", "eggs_laid", "flights", "born_t", "hydration", "dust", "grooming", "steer_base", "last_groom", "last_drink", "coma")
        for c in self._columns:
            shape = {"genome": (0, len(GENES)), "senses": (0, len(SENSES)), "steer_base": (0, len(STEER_TYPES))}.get(c, (0,))
            setattr(self, c, np.zeros(shape))
        self.read = {k: np.zeros(0) for k in READOUT}
        xs, ys = zip(*[self.arena.free_spot() for _ in range(B)])
        self._append(np.array(xs), np.array(ys), np.ones((B, len(GENES))), np.zeros(B), np.full(B, -1))

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
            "bitter": np.array(con.indices(i for i in neurons.BITTER_GRN if i in con.index_of)),
            "water": np.array(con.indices(i for i in neurons.WATER_GRN if i in con.index_of)),
            "jon_ce": np.array(con.indices(i for i in neurons.JON_CE if i in con.index_of)),
            "touch_L": np.flatnonzero((sub == "head bristle") & (side == "left")),
            "touch_R": np.flatnonzero((sub == "head bristle") & (side == "right")),
            "pam": np.flatnonzero(np.char.startswith(ct.astype(str), "PAM")),      # reward dopamine neurons
            "ppl1": np.flatnonzero(np.char.startswith(ct.astype(str), "PPL1")),    # punishment dopamine neurons
        }
        self.group_names = list(groups)
        self.group_gene = {"vis_L": "vision", "vis_R": "vision", "loom_L": "looming", "loom_R": "looming",
                           "sugar": "taste", "bitter": "taste", "water": "taste", "jon_ce": None, "touch_L": None, "touch_R": None,
                           "pam": None, "ppl1": None}
        idx, col = [], []
        for g, name in enumerate(self.group_names):
            idx.extend(groups[name].tolist())
            col.extend([g] * len(groups[name]))
        self.sense_idx = torch.tensor(idx, device=self.dev)
        self.sense_col = torch.tensor(col, device=self.dev)
        read_idx, read_row = [], []
        for r, (name, (t, s)) in enumerate(READOUT.items()):
            sel = [con.index_of[neurons.MN9]] if name == "MN9" else [con.index_of[neurons.ABN1]] if name == "aBN1" else \
                np.flatnonzero((ct == t) & ((side == s) if s else True)).tolist()
            read_idx.extend(sel)
            read_row.extend([r] * len(sel))
        self.read_idx = torch.tensor(read_idx, device=self.dev)
        M = torch.zeros(len(READOUT), len(read_idx), device=self.dev)
        M[torch.tensor(read_row), torch.arange(len(read_idx))] = 1
        self.read_mix = M / M.sum(1, keepdim=True).clamp(min=1)
        self.npf_idx = torch.tensor(np.flatnonzero(np.char.find(ct.astype(str), "NPF") >= 0), device=self.dev)

    # --- population -----------------------------------------------------------
    def _append(self, xs, ys, genomes, generations, parents):
        n = len(xs)
        slots = [self.free_slots.pop(0) for _ in range(n)]
        self._reset_slots(slots)
        new = {c: np.zeros(n) for c in self._columns}
        new["slot"] = np.array(slots, dtype=float)
        new.update({"x": xs, "y": ys, "heading": self.rng.uniform(-np.pi, np.pi, n), "energy": np.full(n, 0.7),
                    "lifespan": self.rng.uniform(*self.cfg.lifespan, n), "jumped_at": np.full(n, -1e9),
                    "last_feed": np.full(n, -1e9), "last_egg": np.full(n, -1e9),
                    "last_groom": np.full(n, -1e9), "last_drink": np.full(n, -1e9), "generation": generations,
                    "parent": parents, "born_t": np.full(n, self.t), "hydration": np.ones(n), "genome": genomes, "senses": np.zeros((n, len(SENSES))),
                    "steer_base": np.zeros((n, len(STEER_TYPES)))})
        for c in self._columns:
            setattr(self, c, np.concatenate([getattr(self, c), new[c]]))
        self.read = {k: np.concatenate([v, np.zeros(n)]) for k, v in self.read.items()}
        born = list(range(self.next_id, self.next_id + n))
        self.ids.extend(born)
        self.next_id += n
        return born

    def _hatch(self):
        # eggs wait for a free brain slot and for warmth (world15 lost 29 of 36 eggs to a full population)
        ready = [e for e in self.eggs if self.t - e.laid >= self.cfg.hatch_time
                 and float(self.arena.temperature(self.t, np.array([e.x]), np.array([e.y]))[0]) >= self.cfg.hatch_min_temp]
        room = len(self.free_slots)
        for e in ready[:max(room, 0)]:
            self.eggs.remove(e)
            genome = e.genome * np.exp(self.rng.normal(0, self.cfg.mutation, len(GENES)))
            fid = self._append(np.array([e.x]), np.array([e.y]), genome[None], np.array([e.generation]),
                               np.array([e.parent]))[0]
            self.births += 1
            self.events.append({"t": round(self.t, 2), "fly": fid, "kind": "born", "parent": int(e.parent),
                                "text": f"вылупилась (поколение {e.generation}, мать №{e.parent})"})
        while len(self.eggs) > self.cfg.max_eggs:
            self.eggs.pop(0)

    # --- one world step -----------------------------------------------------
    def _user_flies(self):
        # flies released by the viewer: founders (genome of ones, generation 0), only into free brain slots
        for x, y in self.arena.pending_flies:
            if not self.free_slots:
                self.events.append({"t": round(self.t, 2), "fly": -1, "kind": "user_cap", "x": round(x), "y": round(y),
                                    "text": f"мух уже {len(self.ids)} — все {self.cfg.max_flies} мозгов заняты"})
                continue
            fid = self._append(np.array([x]), np.array([y]), np.ones((1, len(GENES))), np.zeros(1), np.full(1, -1))[0]
            self.events.append({"t": round(self.t, 2), "fly": fid, "kind": "user_fly", "x": round(x), "y": round(y),
                                "text": "пользователь выпустил муху"})
        self.arena.pending_flies.clear()

    @torch.no_grad()
    def step(self):
        self._user_flies()
        if not self.ids:
            # an empty world keeps living: eggs may still hatch, the viewer may release flies
            self._hatch()
            self.arena.update(self.t, WORLD_DT, {"ids": [], "x": np.zeros(0), "y": np.zeros(0), "stuck": np.zeros(0, int),
                                                 "airborne": np.zeros(0, bool), "moving": np.zeros(0, bool)})
            for e in self.arena.log:
                self.events.append({**e, "fly": -1})
            self.arena.log.clear()
            self.t += WORLD_DT
            return
        light = self.arena.light(self.t)
        slots = torch.tensor(self.slot.astype(np.int64))
        g = torch.tensor(self.genome, dtype=torch.float32)
        self._host_turn ^= 1
        rates = self._rates_host[self._host_turn]
        rates.zero_()
        rates[:len(self.sense_idx), slots] = self._sensory_rates(light)
        n_s, n_o, n_h = len(self.sense_idx), len(self.olf.input_idx), len(self.hyg.input_idx)
        rates[n_s:n_s + n_o] = self._olfaction()
        rates[n_s:n_s + n_o, slots] *= g[:, GENES.index("smell")]
        rates[n_s + n_o:n_s + n_o + n_h] = self._hygro()
        rates[n_s + n_o + n_h:, slots] = self._wind_and_compass()
        bias = self._bias_host[self._host_turn]
        bias.zero_()
        bias[slots] = 4.0 * torch.tensor(np.clip(1 - self.energy, 0, 1), dtype=torch.float32) * g[:, GENES.index("hunger")]
        hz = self._brain_hz(rates, bias, slots)
        a = WORLD_DT / READ_TAU
        for r, k in enumerate(READOUT):
            self.read[k] += (hz[r] - self.read[k]) * a
        self._body(light)
        self._hatch()
        self.arena.update(self.t, WORLD_DT, {"ids": self.ids, "x": self.x, "y": self.y,
                                             "stuck": self.stuck.astype(int), "airborne": self.air_left > 0,
                                             "moving": self.state == 0})
        self._thaw_rescue()
        self._predators()
        for e in self.arena.log:
            self.events.append({**e, "fly": -1})
        self.arena.log.clear()
        self.t += WORLD_DT

    def _brain_hz(self, rates, bias, slots) -> np.ndarray:
        """Run 10 ms of all brains; returns (readouts, flies) descending-neuron rates in Hz."""
        if isinstance(self.brain, FastBrain):
            # CPU and GPU overlap: the body uses the block that finished while the
            # CPU was preparing this step (one extra 10 ms of sensorimotor delay).
            if self._pending is None:
                self._pending_host = torch.zeros(self.brain.counts.shape, pin_memory=True)
                self._overflow_host = torch.zeros(2, pin_memory=True)
            else:
                self._pending.synchronize()
            prev = self._pending_host.clone()
            overflowed = bool((self._overflow_host > 0).any())
            kinds = self._overflow_host.clone()
            self.brain.rates.copy_(rates, non_blocking=True)
            self.brain.bias.copy_(bias, non_blocking=True)
            self.brain.run()
            self._pending_host.copy_(self.brain.counts, non_blocking=True)
            self._overflow_host.copy_(self.brain.overflow_kind, non_blocking=True)
            self._pending = torch.cuda.Event()
            self._pending.record()
            if self.mb is not None:
                k = self.n_read + len(self.mb.kc_idx)
                self.mb.step(self.brain.counts[self.n_read:k], self.brain.counts[k:], WORLD_DT)
            hz = ((self.read_mix_cpu @ prev[:self.n_read, slots]) / WORLD_DT).numpy()
            if overflowed:
                self._pending.synchronize()
                self._check_overflow(bool(kinds[0] > 0), bool(kinds[1] > 0))
            return hz
        counts = torch.zeros(len(self.read_idx), self.cfg.max_flies, device=self.dev)
        rates_d, bias_d = rates.to(self.dev), bias.to(self.dev)
        for _ in range(self.steps_per_world):
            if len(self.npf_idx):
                self.brain.g[self.npf_idx] += bias_d * (self.brain.p.dt / self.brain.p.tau)
            counts += self.brain.step(self.input_idx, rates_d)[self.read_idx]
        if self.mb is not None:
            k = self.n_read + len(self.mb.kc_idx)
            self.mb.step(counts[self.n_read:k], counts[k:], WORLD_DT)
        return ((self.read_mix @ counts[:self.n_read, slots.to(self.dev)]) / WORLD_DT).cpu().numpy()

    def _sensory_rates(self, light: float) -> torch.Tensor:
        cfg, ar, B = self.cfg, self.arena, len(self.ids)
        drive = np.zeros((len(self.group_names), B), dtype=np.float32)
        gi = {n: i for i, n in enumerate(self.group_names)}
        sps, cs, fa, ants = ar.spiders, ar.centipedes, ar.food_arrays(self.t), ar.ants
        # vision: flies, food and flowers, stones, spiders, centipedes, ants as objects in each hemifield
        stones = [o for o in ar.obstacles if o.kind == "stone"]
        ox = np.concatenate([self.x, fa["x"], [o.x for o in stones], [s.x for s in sps], [c.x for c in cs], ants["x"]])
        oy = np.concatenate([self.y, fa["y"], [o.y for o in stones], [s.y for s in sps], [c.y for c in cs], ants["y"]])
        osize = np.concatenate([np.full(B, 1.5), fa["r"], [o.rx for o in stones], [4.0] * len(sps), [10.0 * c.size for c in cs],
                                np.full(len(ants["x"]), 1.0)])
        dx, dy = ox[None] - self.x[:, None], oy[None] - self.y[:, None]
        dist = np.hypot(dx, dy) + 1e-6
        az = np.angle(np.exp(1j * (np.arctan2(dy, dx) - self.heading[:, None])))
        ang = np.clip(2 * osize[None] / dist, 0, 1.5)
        ang[np.arange(B), np.arange(B)] = 0
        ang[dist > 80] = 0
        left = (ang * np.clip(az / 0.5, 0, 1)).sum(1)
        right = (ang * np.clip(-az / 0.5, 0, 1)).sum(1)
        drive[gi["vis_L"]] = 150 * light * np.clip(left / 0.5, 0, 1)
        drive[gi["vis_R"]] = 150 * light * np.clip(right / 0.5, 0, 1)
        # looming: diving birds, the spider walking to a stuck fly, centipedes closing in
        loom = np.zeros((2, B))
        for sh in ar.shadows:
            if sh.target in self.ids:
                i = self.ids.index(sh.target)
                p = min(1.0, (self.t - sh.t_start) / sh.duration)
                rel = np.angle(np.exp(1j * (sh.direction - self.heading[i])))
                strength = 180 * p ** 2 * (0.3 + 0.7 * light)
                loom[0, i] = max(loom[0, i], strength * (1.0 if rel > -0.3 else 0.3))
                loom[1, i] = max(loom[1, i], strength * (1.0 if rel < 0.3 else 0.3))
        for px, py, size, rng_mm in [(s.x, s.y, 4.0, 25.0) for s in sps] + [(c.x, c.y, 10.0 * c.size, 35.0) for c in cs]:
            d = np.hypot(self.x - px, self.y - py)
            close = d < rng_mm
            if close.any():
                rel = np.angle(np.exp(1j * (np.arctan2(py - self.y, px - self.x) - self.heading)))
                strength = 180 * np.clip(1 - d / rng_mm, 0, 1) ** 2 * (0.4 + 0.6 * light)
                loom[0] = np.where(close, np.maximum(loom[0], strength * np.where(rel > -0.3, 1.0, 0.3)), loom[0])
                loom[1] = np.where(close, np.maximum(loom[1], strength * np.where(rel < 0.3, 1.0, 0.3)), loom[1])
        drive[gi["loom_L"]], drive[gi["loom_R"]] = loom[0], loom[1]
        # taste on contact with food, stronger when hungry
        on = self._touching_food(fa)
        best = np.where(on, fa["taste"][None], 0).max(1) if len(fa["x"]) else np.zeros(B)
        sugar = 150 * (0.5 + np.clip(1 - self.energy, 0, 1)) * best
        drive[gi["sugar"]] = sugar
        bitter = 200 * (np.where(on, fa["bitter"][None], 0).max(1) if len(fa["x"]) else np.zeros(B))
        drive[gi["bitter"]] = bitter
        # water at a pond edge (not through ice), stronger when thirsty; dust on the antennae
        water = 150 + 250 * np.clip(1 - self.hydration, 0, 1)
        at_water = self._at_water()
        drive[gi["water"]] = np.where(at_water, water, 0)
        drive[gi["jon_ce"]] = 200 * np.clip(self.dust, 0, 1) * (self.state != 3)
        if self.mb is not None:
            # what the connectome does not do by itself: reward and punishment reach the dopamine neurons
            drive[gi["pam"]] = cfg.dan_hz * np.clip(sugar / 150, 0, 1)
            drive[gi["ppl1"]] = cfg.dan_hz * np.maximum(np.maximum(np.clip(bitter / 200, 0, 1), (self.stuck > 0) * 1.0),
                                                        np.clip((loom.max(0) / 180 - 0.5) * 2, 0, 1))
        self.senses[:, 8] = bitter / 200
        self.senses[:, 9] = at_water * water / 400
        self.senses[:, 10] = np.clip(self.dust, 0, 1)
        # touch: head bristles pressing against a wall, stone or water edge
        drive[gi["touch_L"], (self.touch_left > 0) & (self.state != 3)] = 100
        drive[gi["touch_R"], (self.touch_left < 0) & (self.state != 3)] = 100
        for name, gene in self.group_gene.items():
            if gene:
                drive[gi[name]] *= self.genome[:, GENES.index(gene)]
        self.senses[:, 5] = loom.max(0) / 180
        self.senses[:, 6] = self.touch_left != 0
        self.senses[:, 7] = sugar / 150
        return torch.from_numpy(drive)[self.sense_col_cpu]

    def _at_water(self) -> np.ndarray:
        """Walking fly whose head reaches open water (a pond edge)."""
        ar = self.arena
        if ar.frozen or not len(self.ids):
            return np.zeros(len(self.ids), dtype=bool)
        hx, hy = self.x + 2.0 * np.cos(self.heading), self.y + 2.0 * np.sin(self.heading)
        return ar.in_water(hx, hy) & (self.air_left <= 0) & (self.stuck == 0)

    def _touching_food(self, fa) -> np.ndarray:
        """(flies, food) contact: the fly's head is over an edible item."""
        if not len(fa["x"]):
            return np.zeros((len(self.ids), 0), dtype=bool)
        d = np.hypot(self.x[:, None] - fa["x"][None], self.y[:, None] - fa["y"][None])
        return (d < fa["r"][None] + 0.5) & fa["edible"][None]

    def _wind_and_compass(self) -> torch.Tensor:
        # wind on the antennae (Johnston's organ) and head direction on the E-PG ring; no wind sense in the air
        wx, wy = (float(v) for v in self.arena.wind)
        rel = np.angle(np.exp(1j * (np.arctan2(-wy, -wx) - self.heading)))          # where the wind comes from, + = left
        speed = np.where(self.air_left > 0, 0.0, np.hypot(wx, wy))
        wind = self.wind_sense.rates(torch.tensor(speed, dtype=torch.float32), torch.tensor(rel, dtype=torch.float32))
        heading = self.compass.rates(torch.tensor(self.heading, dtype=torch.float32))
        return torch.cat([wind, heading])

    def _olfaction(self) -> torch.Tensor:
        B, ar = len(self.ids), self.arena
        conc = np.zeros((B, len(self.olf.names), 2), dtype=np.float32)
        for s, sign in ((0, 1), (1, -1)):
            ax = self.x + 1.2 * np.cos(self.heading) - sign * 0.8 * np.sin(self.heading)
            ay = self.y + 1.2 * np.sin(self.heading) + sign * 0.8 * np.cos(self.heading)
            for o, name in enumerate(self.olf.names):
                if name == "fly":
                    d2 = (ax[:, None] - self.x[None]) ** 2 + (ay[:, None] - self.y[None]) ** 2
                    np.fill_diagonal(d2, np.inf)
                    conc[:, o, s] = np.exp(-d2 / (2 * 6.0 ** 2)).sum(1)
                elif name == "spider":
                    conc[:, o, s] = ar.odor("spider", ax, ay, self.t) + ar.odor("centipede", ax, ay, self.t)
                else:
                    conc[:, o, s] = ar.odor(name, ax, ay, self.t)
        n = self.olf.names
        self.senses[:, 0:2] = conc[:, n.index("fruit"), :]
        self.senses[:, 2:4] = conc[:, n.index("vinegar"), :]
        self.senses[:, 4] = conc[:, n.index("spider"), :].max(1)
        full = np.zeros((self.cfg.max_flies, *conc.shape[1:]), dtype=np.float32)
        full[self.slot.astype(np.int64)] = conc
        return self.olf.rates(torch.from_numpy(full), WORLD_DT * 1000)

    def _hygro(self) -> torch.Tensor:
        """Humidity at the antennae -> sacculus moist cells. Thirst raises the gain: in the fly thirst turns
        humid air attractive; here the sign of the response is the connectome's, only the gain is ours."""
        B, ar = len(self.ids), self.arena
        conc = np.zeros((B, len(self.hyg.names), 2), dtype=np.float32)
        gain = self.cfg.humid_gain[0] + self.cfg.humid_gain[1] * np.clip(1 - self.hydration, 0, 1)
        k = self.hyg.names.index("humid")
        for s, sign in ((0, 1), (1, -1)):
            ax = self.x + 1.2 * np.cos(self.heading) - sign * 0.8 * np.sin(self.heading)
            ay = self.y + 1.2 * np.sin(self.heading) + sign * 0.8 * np.cos(self.heading)
            conc[:, k, s] = ar.humidity(ax, ay) * gain
        self.senses[:, 11] = conc[:, k, :].max(1)
        full = np.zeros((self.cfg.max_flies, *conc.shape[1:]), dtype=np.float32)
        full[self.slot.astype(np.int64)] = conc
        return self.hyg.rates(torch.from_numpy(full), WORLD_DT * 1000)

    def _body(self, light: float):
        cfg, dt, ar = self.cfg, WORLD_DT, self.arena
        r = self.read
        B = len(self.ids)
        if cfg.steer is None:
            diff = (r["DNa01 L"] + r["DNa02 L"]) - (r["DNa01 R"] + r["DNa02 R"])
            self.turn_base += (diff - self.turn_base) * (dt / cfg.turn_adapt)
            turn = cfg.turn_gain * (diff - self.turn_base)
        else:
            diffs = np.stack([r[f"{t} L"] - r[f"{t} R"] for t in STEER_TYPES], 1)          # (flies, types)
            self.steer_base += (diffs - self.steer_base) * (dt / cfg.turn_adapt)
            w = np.array([cfg.steer.get(t, 0.0) for t in STEER_TYPES])
            turn = np.clip((diffs - self.steer_base) @ w / 50.0 + cfg.steer.get("bias", 0.0), -4.0, 4.0)
        airborne = self.air_left > 0
        fa = ar.food_arrays(self.t)
        on_fruit = self._touching_food(fa).any(1)
        stuck = self.stuck > 0
        # cold: slower walking, chill coma (no walking, takeoff, feeding) with 1 deg hysteresis
        temp = self.fly_temperature()
        coma = ~airborne & np.where(self.coma > 0, temp < cfg.chill_temp + 1.0, temp < cfg.chill_temp)
        for i in np.flatnonzero(coma & (self.coma == 0)):
            self._event(i, "chill_coma", f"оцепенела от холода ({temp[i]:.0f} °C)")
        for i in np.flatnonzero(~coma & (self.coma > 0)):
            self._event(i, "chill_wake", "отогрелась")
        self.coma = coma.astype(float)
        warmth = np.clip((temp - cfg.chill_temp) / (cfg.warm_temp - cfg.chill_temp), 0, 1)
        feeding = on_fruit & (r["MN9"] > cfg.feed_threshold) & ~airborne & ~stuck & ~coma
        backward = (r["MDN"] > cfg.mdn_threshold) & ~airborne & ~stuck & ~coma
        drinking = self._at_water() & ~on_fruit & (r["MN9"] > cfg.feed_threshold) & ~airborne & ~stuck & ~coma
        busy = feeding | drinking | airborne | stuck | coma
        was_grooming = self.grooming > 0
        self.grooming = np.where(busy, 0, np.where(self.grooming > 0, self.dust > 0.08,
                                                   r["aBN1"] > cfg.groom_threshold)).astype(float)
        grooming = self.grooming > 0
        for i in np.flatnonzero(grooming & (self.t - self.last_groom > 5.0)):
            self._event(i, "groom", "чистит антенны (JON-CE → aBN1)")
        for i in np.flatnonzero(drinking & (self.t - self.last_drink > 5.0)):
            self._event(i, "drink", "пьёт воду (водяные рецепторы → MN9)")
        self.last_groom[grooming] = self.t
        self.last_drink[drinking] = self.t
        self.cooldown = np.maximum(self.cooldown - dt, 0)
        ready = (self.cooldown <= 0) & ~airborne
        long_drive = (r["DNp02"] + r["DNp04"] + r["DNp11"]) / 3
        hop = ready & (r["GF"] > cfg.gf_threshold)
        fly_long = ready & ~hop & (long_drive > cfg.long_threshold)
        hop &= ~coma
        fly_long &= ~coma

        # a stuck fly struggles on its own; takeoff attempts of the brain add escape force
        self.escape_force = np.maximum(self.escape_force - cfg.escape_decay * dt, 0)
        self.escape_force[stuck] += cfg.struggle * dt
        for i in np.flatnonzero(stuck & (hop | fly_long)):
            self.escape_force[i] += 0.6 if hop[i] else 0.35
            self.cooldown[i] = 0.4
            self._event(i, "web_struggle", "бьётся в паутине")
        for i in np.flatnonzero(stuck):
            web = ar.web(int(self.stuck[i]) - 1)
            need = cfg.web_escape * (0.4 + 0.6 * (web.strength if web else 0))    # old webs hold less
            if self.escape_force[i] >= need:
                ar.tear_web(int(self.stuck[i]) - 1, self.t)
                self.stuck[i] = 0
                self._event(i, "web_free", "вырвалась из паутины")
        hop &= self.stuck == 0
        fly_long &= self.stuck == 0

        for i in np.flatnonzero(hop | fly_long):
            dur, spd = cfg.hop if hop[i] else cfg.flight
            self.air_left[i] = self.air_total[i] = dur
            self.flights[i] += 1
            self.air_speed[i] = spd
            self.air_height[i] = 0.4 if hop[i] else 1.0
            self.cooldown[i] = dur + 0.8
            self.jumped_at[i] = self.t
            if hop[i]:
                self.heading[i] += self.rng.uniform(-1.0, 1.0)
            self.energy[i] -= cfg.jump_cost if hop[i] else cfg.flight_cost
            self.air_x0[i], self.air_y0[i], self.air_water[i], self.air_long[i] = self.x[i], self.y[i], 0, 0 if hop[i] else 1
            self.counters["hops" if hop[i] else "flights"] += 1
            self._event(i, "hop" if hop[i] else "flight",
                        "прыжок (гигантское волокно)" if hop[i] else "взлетела (DNp02/04/11, долгий перелёт)")
        airborne = self.air_left > 0

        walk = cfg.walk_speed * self.genome[:, GENES.index("walk")]
        speed = walk * (0.35 + 0.65 * light) * (0.3 + 0.7 * warmth)
        speed[feeding | stuck | drinking | grooming | coma] = 0
        speed[backward] = -0.5 * walk[backward]
        speed[airborne] = self.air_speed[airborne]
        walking = ~airborne & ~stuck & ~coma
        self.heading[walking] += (turn * dt + cfg.wander * np.sqrt(dt) * self.rng.normal(0, 1, B))[walking]
        self.heading[stuck] += self.rng.normal(0, 0.15, stuck.sum())       # struggling
        # flight: straight segments joined by body saccades. The brain's turn command (the same DN mapping as
        # on the ground) accumulates as intent and is spent in a saccade once it is large enough; spontaneous
        # saccades come on top. Hops stay straight.
        flying = airborne & (self.air_height >= 1.0)
        self.turn_intent = np.where(flying, self.turn_intent + cfg.air_steer * turn * dt, 0.0)
        fire = flying & (np.abs(self.turn_intent) >= cfg.saccade_threshold)
        self.heading[fire] += np.clip(self.turn_intent[fire], -cfg.saccade_angle[1], cfg.saccade_angle[1])
        self.turn_intent[fire] = 0.0
        sacc = flying & ~fire & (self.rng.random(B) < cfg.saccade_rate * dt)
        n_sacc = int(sacc.sum())
        if n_sacc:
            self.heading[sacc] += self.rng.choice([-1.0, 1.0], n_sacc) * self.rng.uniform(*cfg.saccade_angle, n_sacc)

        W, H = ar.cfg.width, ar.cfg.height
        nx = self.x + speed * np.cos(self.heading) * dt
        ny = self.y + speed * np.sin(self.heading) * dt
        self.touch_left = np.where(self.touch_left != 0, np.sign(self.touch_left) * np.maximum(np.abs(self.touch_left) - dt, 0), 0)
        edge = (nx < 1) | (nx > W - 1) | (ny < 1) | (ny > H - 1)
        blocked = ~airborne & ar.walk_blocked(nx, ny)
        for i in np.flatnonzero((edge | blocked) & ~airborne):
            # walking into an obstacle: head bristles touch it; the body slides along if it can
            ahead = np.angle(np.exp(1j * (np.arctan2(ny[i] - self.y[i], nx[i] - self.x[i]) - self.heading[i])))
            self.touch_left[i] = 0.2 if ahead >= 0 else -0.2
            moved = False
            for rot in (0.9, -0.9, 1.8, -1.8):
                h = self.heading[i] + rot
                tx, ty = self.x[i] + abs(speed[i]) * 0.6 * np.cos(h) * dt, self.y[i] + abs(speed[i]) * 0.6 * np.sin(h) * dt
                if 1 < tx < W - 1 and 1 < ty < H - 1 and not ar.walk_blocked(tx, ty):
                    nx[i], ny[i], moved = tx, ty, True
                    break
            if not moved:
                nx[i], ny[i] = self.x[i], self.y[i]
            if edge[i]:
                self.heading[i] = np.arctan2(H / 2 - self.y[i], W / 2 - self.x[i]) + self.rng.uniform(-0.8, 0.8)
        # flying off the map edge turns the flight back
        out = airborne & ((nx < 2) | (nx > W - 2) | (ny < 2) | (ny > H - 2))
        self.heading[out] += np.pi
        self.x, self.y = np.clip(nx, 1, W - 1), np.clip(ny, 1, H - 1)

        # landing: never on water or a stone — the flight continues until clear ground
        if airborne.any():
            self.air_water[airborne] = np.maximum(self.air_water[airborne], ar.in_water(self.x[airborne], self.y[airborne]))
        landing = airborne & (self.air_left - dt <= 0)
        extend = landing & ar.walk_blocked(self.x, self.y, pad=1)
        self.air_left = np.where(extend, 0.1, np.maximum(self.air_left - dt, 0))
        self.air_total = np.where(extend, self.air_total + 0.1, self.air_total)
        airborne = self.air_left > 0
        self._landing_metrics(landing & ~extend)

        # webs that decayed away release their flies; entering a web
        if (self.stuck > 0).any():
            alive = np.isin(self.stuck - 1, [w.id for w in ar.webs])
            for i in np.flatnonzero((self.stuck > 0) & ~alive):
                self.stuck[i] = 0
                self._event(i, "web_released", "паутина истлела — муха свободна")
        web_at = ar.in_web(self.x, self.y)
        caught = ~airborne & (self.stuck == 0) & (web_at > 0) & (self.t - self.jumped_at > 3.0)
        for i in np.flatnonzero(caught):
            self.stuck[i] = web_at[i]
            self.escape_force[i] = 0
            self._event(i, "web_stuck", "прилипла к паутине")

        # eating: each feeding fly eats from the nearest item it touches; items are finite
        if feeding.any() and len(fa["x"]):
            touch = self._touching_food(fa)
            d = np.where(touch, np.hypot(self.x[:, None] - fa["x"][None], self.y[:, None] - fa["y"][None]), np.inf)
            which = np.argmin(d, 1)
            eaters = np.flatnonzero(feeding & touch.any(1))
            want = np.bincount(which[eaters], minlength=len(ar.food)) * cfg.sugar_per_s * dt
            got = np.minimum(want, np.maximum(fa["amount"], 0))
            share = np.divide(got, want, out=np.zeros_like(got), where=want > 0)
            for k in np.flatnonzero(got > 0):
                ar.food[k].amount -= got[k]
            self.energy[eaters] += cfg.sugar_per_s * dt * share[which[eaters]] * cfg.energy_per_sugar
            self.meals[eaters] += cfg.sugar_per_s * dt * share[which[eaters]]
            self.counters["eaten"] += float(got.sum())
            # pollen sticks to a fly feeding on a flower; the next different flower gets pollinated
            on_flower = eaters[fa["kind"][which[eaters]] == 4]
            if len(on_flower):
                flower_id = np.array([ar.food[k].id for k in which[on_flower]])
                carrying = (self.pollen[on_flower] > 0) & (self.t - self.pollen_t[on_flower] < ar.cfg.pollen_life)
                for i, fl in zip(on_flower[carrying & (self.pollen[on_flower] - 1 != flower_id)],
                                 flower_id[carrying & (self.pollen[on_flower] - 1 != flower_id)]):
                    seeded = ar.pollinated(self.t, int(fl), int(self.pollen[i]) - 1, self.x[i], self.y[i])
                    self._event(i, "pollinated", "опылила цветок" + (" — завяжется семя" if seeded else ""))
                self.pollen[on_flower] = flower_id + 1
                self.pollen_t[on_flower] = self.t
        self.pollen[self.t - self.pollen_t > ar.cfg.pollen_life] = 0
        for i in np.flatnonzero(feeding & (self.t - self.last_feed > 3.0)):
            self._event(i, "feed", "ест (MN9 → хоботок)")
        self.last_feed[feeding] = self.t
        can_lay = feeding & (self.age > cfg.maturity) & (self.t - self.last_egg > cfg.egg_interval) & (self.energy > cfg.egg_min_energy)
        for i in np.flatnonzero(can_lay):
            self.eggs.append(Egg(float(self.x[i]), float(self.y[i]), self.t, self.ids[i],
                                 int(self.generation[i]) + 1, self.genome[i].copy()))
            self.energy[i] -= cfg.egg_energy
            self.last_egg[i] = self.t
            self.eggs_laid[i] += 1
            self._event(i, "egg", "отложила яйцо")
        # thirst, drinking, juice from food; dust from the ground, litter and flowers; grooming cleans it (and the pollen)
        self.hydration -= dt / cfg.thirst_s * np.where(coma, cfg.coma_metabolism, 1.0)   # a torpid fly loses little water
        self.hydration[drinking] += cfg.drink_per_s * dt
        walking_now = ~airborne & ~stuck & (np.abs(speed) > 0)
        self.dust += dt * walking_now * np.where(ar.in_litter(self.x, self.y) > 0, cfg.dust_litter, cfg.dust_ground)
        if feeding.any() and len(fa["x"]):
            self.hydration[eaters] += cfg.juice * cfg.sugar_per_s * dt * share[which[eaters]]
            self.dust[on_flower] += cfg.dust_flower * dt
        self.dust[grooming] -= cfg.groom_clean * dt
        self.dust = np.clip(self.dust, 0, 1)
        cleaned = was_grooming & ~grooming & (self.pollen > 0) & (self.rng.random(len(self.ids)) < 0.5)   # a bout ends
        for i in np.flatnonzero(cleaned):
            self._event(i, "pollen_groomed", "счистила пыльцу")
        self.pollen[cleaned] = 0
        self.hydration = np.clip(self.hydration, 0, 1)
        # 0 walk, 1 feed, 2 backward, 3 airborne, 4 stuck in web, 5 drink, 6 groom, 7 chill coma
        self.state = np.where(airborne, 3, np.where(self.stuck > 0, 4, np.where(feeding, 1, np.where(drinking, 5,
                              np.where(grooming, 6, np.where(coma, 7, np.where(backward, 2, 0)))))))

        self.energy -= cfg.metabolism * dt * np.where(coma, cfg.coma_metabolism, 0.5 + 0.5 * warmth) \
            + cfg.move_cost * np.abs(speed) * dt * (~airborne)
        self.energy = np.clip(self.energy, 0, 1)
        self.age += dt

        for sh in list(ar.shadows):
            if sh.target in self.ids and self.t - sh.t_start >= sh.duration and not sh.resolved:
                i = self.ids.index(sh.target)
                sh.resolved = True
                if self.jumped_at[i] >= sh.t_start:
                    self._event(i, "escape", "увернулась от птицы")
                else:
                    self._kill(i, "bird", "поймана птицей")
        for i in np.flatnonzero(self.energy <= 0):
            self._kill(i, "starved", "умерла от голода")
        for i in np.flatnonzero(self.hydration <= 0):
            self._kill(i, "thirst", "умерла от жажды")
        for i in np.flatnonzero(self.age >= self.lifespan):
            self._kill(i, "old", "умерла от старости")
        self._remove_dead()

    def _landing_metrics(self, landed):
        for i in np.flatnonzero(landed):
            dist = float(np.hypot(self.x[i] - self.air_x0[i], self.y[i] - self.air_y0[i]))
            if self.air_long[i] and dist >= LONG_FLIGHT_MM:
                self.counters["long_flights"] += 1
                self._event(i, "long_flight", f"долгий перелёт: {dist:.0f} мм")
            if self.air_water[i] and not self.arena.in_water(self.air_x0[i], self.air_y0[i]) \
                    and not self.arena.in_water(self.x[i], self.y[i]):
                self.counters["water_crossings"] += 1
                self._event(i, "water_crossed", f"перелетела через воду ({dist:.0f} мм)")
            self.air_water[i] = 0
            if self.air_long[i]:
                fa = self.arena.food_arrays(self.t)
                if len(fa["x"]) and float(np.min(np.hypot(fa["x"] - self.x[i], fa["y"] - self.y[i]) - fa["r"])) < 15.0:
                    self.counters["landings_by_food"] += 1
                    self._event(i, "landed_by_food", "села рядом с едой")

    def _thaw_rescue(self):
        # when ice melts under walking flies, the body is put on the nearest shore (world rule, no drowning)
        ar = self.arena
        if ar.frozen or not len(self.ids):
            return
        wet = (self.air_left <= 0) & ar.in_water(self.x, self.y)
        for i in np.flatnonzero(wet):
            self.x[i], self.y[i] = ar.near_free(float(self.x[i]), float(self.y[i]))
            self._event(i, "ice_shore", "лёд растаял под ногами — выбралась на берег")

    def _predators(self):
        ar = self.arena
        dead = {d for d, _ in self.dead}
        for s in ar.spiders:
            if s.target is not None and s.target in self.ids and s.target not in dead:
                i = self.ids.index(s.target)
                if self.stuck[i] and np.hypot(self.x[i] - s.x, self.y[i] - s.y) < 2.5:
                    self._kill(i, "spider", "съедена пауком")
                    ar.predator_ate(s, self.t)
                    s.target = None
                    dead.add(self.ids[i])
        for c in ar.centipedes:
            if c.target is not None and c.target in self.ids and c.target not in dead:
                i = self.ids.index(c.target)
                if self.air_left[i] <= 0 and np.hypot(self.x[i] - c.x, self.y[i] - c.y) < 3.5:
                    self._kill(i, "centipede", "поймана сороконожкой")
                    ar.predator_ate(c, self.t)
                    c.target = None
        self._remove_dead()

    # --- bookkeeping --------------------------------------------------------
    def _event(self, i: int, kind: str, text: str):
        self.events.append({"t": round(self.t, 2), "fly": self.ids[i], "kind": kind, "text": text,
                            "x": round(float(self.x[i])), "y": round(float(self.y[i]))})

    def _kill(self, i: int, kind: str, text: str):
        if self.ids[i] not in {d for d, _ in self.dead}:
            self._event(i, "death_" + kind, text)
            self.events[-1].update({   # obituary facts for the viewer
                "age": round(float(self.age[i]), 1), "generation": int(self.generation[i]), "parent": int(self.parent[i]),
                "energy": round(float(self.energy[i]), 2), "hydration": round(float(self.hydration[i]), 2), "meals": round(float(self.meals[i]), 1),
                "eggs": int(self.eggs_laid[i]), "flights": int(self.flights[i]),
                "since_meal": round(float(self.t - self.last_feed[i]), 1) if self.last_feed[i] > -1e8 else None,
                "genome": [round(float(v), 2) for v in self.genome[i]]})
            self.dead.append((self.ids[i], kind))
            self.arena.fly_died(self.t, self.ids[i], kind, float(self.x[i]), float(self.y[i]))

    def _remove_dead(self):
        gone = {d for d, _ in self.dead}
        keep = np.array([fid not in gone for fid in self.ids])
        if keep.all():
            return
        cols = np.flatnonzero(keep)
        freed = [int(s) for s in self.slot[~keep]]
        self._reset_slots(freed)
        self.free_slots.extend(freed)
        for c in self._columns:
            setattr(self, c, getattr(self, c)[cols])
        self.read = {k: v[cols] for k, v in self.read.items()}
        self.ids = [fid for fid, k in zip(self.ids, keep) if k]

    def _reset_slots(self, slots):
        if not slots:
            return
        if isinstance(self.brain, FastBrain):
            self.brain.reset_columns(slots)
        else:
            mask = torch.zeros(self.cfg.max_flies, dtype=torch.bool, device=self.dev)
            mask[slots] = True
            self.brain.reset(mask)
        self.olf.state[slots] = 0
        self.hyg.state[slots] = 0
        if self.mb is not None:
            self.mb.reset(slots)

    def _check_overflow(self, events: bool = True, spikes: bool = True):
        # the graph path drops spikes if its fixed buffers overflow: grow the one that overflowed and re-record
        # (each block costs about the size of the event buffer, so it grows gently and only when it was full)
        if isinstance(self.brain, FastBrain) and (events or spikes):
            b = self.brain
            if events:
                b.max_events = int(b.max_events * 1.25)
            if spikes:
                b.max_spikes = int(b.max_spikes * 1.25)
            b._alloc_events()
            b.overflow.zero_()
            b.overflow_kind.zero_()
            state = [t.clone() for t in (b.v, b.g, b.refrac, b.spike_buf, b.m)]
            b.capture()
            for dst, src in zip((b.v, b.g, b.refrac, b.spike_buf, b.m), state):
                dst.copy_(src)
            self.events.append({"t": round(self.t, 2), "fly": -1, "kind": "info", "x": None, "y": None,
                                "text": f"буферы мозга увеличены до {b.max_spikes} спайков / {b.max_events} событий"})

    def fly_temperature(self) -> np.ndarray:
        """Local temperature (deg C) at each fly. HOOK for future physiology (cold walking, chill coma,
        diapause, thermosensation): nothing in the body or the brain reads it yet."""
        return np.asarray(self.arena.temperature(self.t, self.x, self.y), dtype=float).reshape(len(self.ids))

    def memory(self) -> dict:
        """Mean plastic multiplier per MBON type over the living flies (1 = nothing learned), and per fly
        the memory strength 1 - mean multiplier."""
        if self.mb is None or not len(self.ids):
            return {"types": {}, "flies": []}
        cols = torch.as_tensor(self.slot.astype(np.int64), device=self.dev)
        m = self.brain.m[:, cols]
        return {"types": {t: round(v, 3) for t, v in self.mb.summary(cols).items()},
                "flies": [round(float(v), 3) for v in (1 - m.mean(0)).cpu()]}

    def frame(self) -> dict:
        temp = self.fly_temperature()
        alt =np.where(self.air_total > 0, np.sin(np.pi * np.clip(1 - self.air_left / np.maximum(self.air_total, 1e-6), 0, 1)), 0)
        alt = np.where(self.air_left > 0, alt * self.air_height, 0)
        return {
            **self.arena.snapshot(self.t),
            "counters": {**self.counters, "eaten": round(self.counters["eaten"]),
                         **{k: round(v) for k, v in self.arena.counters.items()}},
            "eggs": [[round(e.x, 1), round(e.y, 1), round((self.t - e.laid) / self.cfg.hatch_time, 2), e.parent]
                     for e in self.eggs],
            "flies": [[fid, round(float(self.x[i]), 1), round(float(self.y[i]), 1), round(float(self.heading[i]), 2),
                       int(self.state[i]), round(float(self.energy[i]), 3), round(float(self.age[i]), 1),
                       [int(self.read[k][i]) for k in READOUT], int(self.generation[i]), int(self.parent[i]),
                       [round(float(v), 2) for v in self.genome[i]], round(float(alt[i]), 2),
                       [round(float(v), 2) for v in self.senses[i]], round(float(self.escape_force[i]), 2),
                       int(self.pollen[i] > 0), round(float(temp[i]), 1), round(float(self.hydration[i]), 3),
                       round(float(self.dust[i]), 2)]
                      for i, fid in enumerate(self.ids)],
        }
