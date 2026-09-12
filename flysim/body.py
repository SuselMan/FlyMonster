"""Wiring flies to the maze: which neurons receive senses, which are read out.

Each sensory feature (e.g. "odor at left antenna", "trap ahead") drives its
own small, disjoint group of real sensory neurons of the matching type.
Actions and message symbols are read from pooled descending neurons.
"""
from dataclasses import dataclass

import numpy as np
import torch

from . import config, connectome

N_SYMBOLS = 4
MAX_SLOTS = 2   # symbols per message
RAYS = ("fwd", "left", "right", "back")


def hear_name(slot: int, k: int) -> str:
    return f"hear_{k}" if slot == 0 else f"hear{slot + 1}_{k}"


# Hearing features go last, slot by slot, so older parameter vectors stay a prefix.
HEAR_FEATURES = [hear_name(s, k) for s in range(MAX_SLOTS) for k in range(N_SYMBOLS)]
NOSE_FEATURES = ["smell_L", "smell_R", "smell_more_L", "smell_more_R", "bump"] + HEAR_FEATURES
EYE_FEATURES = [f"{ray}_{what}" for ray in RAYS for what in ("open", "trap", "food")] + HEAR_FEATURES


@dataclass
class Wiring:
    groups: dict[str, dict[str, list[int]]]   # role -> feature -> neuron indices
    readout: np.ndarray                         # descending neuron indices
    pool_of: np.ndarray                         # pool id per readout neuron
    n_pools: int


def build(con: connectome.Connectome, group_size: int = 30, n_pools: int = 32,
          seed: int = 0) -> Wiring:
    ann = connectome.annotations()
    ann = ann[~ann.index.duplicated() & ann.index.isin(list(con.index_of))]
    rng = np.random.default_rng(seed)
    dn = np.array(con.indices(ann.index[ann.super_class == "descending"]))
    reach = dn_reach(con, dn)

    def pool(mask, ranked=False, safe=False) -> list[int]:
        idx = np.array(con.indices(ann.index[mask]))
        if safe:
            idx = idx[~np.isin(idx, runaway_neurons(con, idx))]
        if ranked:  # neurons with the strongest excitatory path to descending neurons first
            return list(idx[np.argsort(-reach[idx], kind="stable")])
        return list(rng.permutation(idx))

    # Anything entering the antennal lobe (olfactory, hygro-, thermosensory)
    # drives this model into self-sustained runaway activity in the AL/mushroom
    # body loop, with no left/right difference left. Gustatory neurons are the
    # chemosensory entry that stays stable and lateralized, so "smell" uses them.
    chem = ann.cell_class == "gustatory"
    vpn = ann.super_class == "visual_projection"
    pools = {
        "chem_L": pool(chem & (ann.side == "left"), ranked=True, safe=True),
        "chem_R": pool(chem & (ann.side == "right"), ranked=True, safe=True),
        "bristle": pool((ann.cell_class == "mechanosensory") & (ann.cell_sub_class == "head bristle")),
        "auditory": pool((ann.cell_class == "mechanosensory")
                         & ann.cell_sub_class.isin(["auditory", "wind_gravity"]), ranked=True),
        # Photoreceptors barely reach the central brain in this model (no tonic
        # activity to modulate), so vision enters at visual projection neurons.
        "eye_L": pool(vpn & (ann.side == "left"), ranked=True),
        "eye_R": pool(vpn & (ann.side == "right"), ranked=True),
        "ocellar": pool((ann.cell_class == "visual") & (ann.cell_sub_class == "ocellar"), ranked=True),
    }

    def take(name, k=group_size):
        out, pools[name] = pools[name][:k], pools[name][k:]
        if len(out) < k:
            raise ValueError(f"not enough neurons left in {name}")
        return [int(i) for i in out]

    nose = {
        "smell_L": take("chem_L"), "smell_R": take("chem_R"),
        "smell_more_L": take("chem_L"), "smell_more_R": take("chem_R"),
        "bump": take("bristle"),
    }
    eye = {}
    for what in ("open", "trap", "food"):
        eye[f"fwd_{what}"] = take("eye_L", group_size // 2) + take("eye_R", group_size // 2)
        eye[f"left_{what}"] = take("eye_L")
        eye[f"right_{what}"] = take("eye_R")
        eye[f"back_{what}"] = take("ocellar", 20)
    for slot in range(MAX_SLOTS):
        for k in range(N_SYMBOLS):
            # The same auditory neurons mean the same symbol for both flies.
            ears = take("auditory", 20)
            nose[hear_name(slot, k)] = ears
            eye[hear_name(slot, k)] = ears

    pool_of = rng.permutation(np.arange(len(dn)) % n_pools)
    return Wiring({"nose": nose, "eye": eye}, dn, pool_of, n_pools)


def runaway_neurons(con: connectome.Connectome, candidates: np.ndarray, chunk: int = 5,
                    threshold: float = 20.0) -> np.ndarray:
    """Candidates whose stimulation leaves the brain in self-sustained activity.

    Drives chunks of neurons at 200 Hz for 200 ms, then checks spikes per step
    during 100 ms without input. Cached per candidate set.
    """
    import hashlib
    import json

    from .brain import FlyBrain
    from .config import LIFParams

    key = hashlib.sha1(np.sort(candidates).tobytes()).hexdigest()[:12]
    cache = config.DATA_CACHE / f"runaway_{key}.json"
    if cache.exists():
        return np.array(json.loads(cache.read_text()), dtype=np.int64)

    chunks = [candidates[i:i + chunk] for i in range(0, len(candidates), chunk)]
    brain = FlyBrain(con, batch=len(chunks), params=LIFParams(dt=0.5))
    idx = torch.tensor(candidates, device=brain.device)
    rates = torch.zeros(len(candidates), len(chunks), device=brain.device)
    for j in range(len(chunks)):
        rates[j * chunk:j * chunk + len(chunks[j]), j] = 200.0
    for _ in range(400):
        brain.step(idx, rates)
    after = torch.zeros(len(chunks), device=brain.device)
    for _ in range(200):
        after += brain.step().sum(0)
    bad = [int(n) for j, c in enumerate(chunks) if after[j] / 200 > threshold for n in c]
    config.DATA_CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(bad))
    return np.array(bad, dtype=np.int64)


def dn_reach(con: connectome.Connectome, dn: np.ndarray) -> np.ndarray:
    """Excitatory synapses onto descending neurons, directly or via one hop."""
    w = con.weights
    w_pos = torch.sparse_csr_tensor(w.crow_indices(), w.col_indices(), w.values().clamp(min=0), w.shape)
    target = torch.zeros(con.n, 1)
    target[dn] = 1.0
    one_hop = torch.sparse.mm(w_pos, target)
    two_hop = torch.sparse.mm(w_pos, one_hop.clamp(max=50))
    return (one_hop + 0.02 * two_hop).squeeze(1).numpy()


class IO:
    """Tensors for feeding features into a batch of brains and reading pools out."""

    def __init__(self, wiring: Wiring, device):
        self.wiring = wiring
        self.device = device
        neurons = sorted({i for role in wiring.groups.values() for g in role.values() for i in g})
        self.input_idx = torch.tensor(neurons, device=device)
        row = {n: r for r, n in enumerate(neurons)}
        # role -> (n_input_neurons, n_features) 0/1 matrix
        self.maps = {}
        for role, names in (("nose", NOSE_FEATURES), ("eye", EYE_FEATURES)):
            m = torch.zeros(len(neurons), len(names), device=device)
            for f, name in enumerate(names):
                m[[row[i] for i in wiring.groups[role][name]], f] = 1.0
            self.maps[role] = m
        self.readout_idx = torch.tensor(wiring.readout, device=device)
        pool_of = torch.tensor(wiring.pool_of, device=device)
        self.pool_matrix = torch.zeros(wiring.n_pools, len(wiring.readout), device=device)
        self.pool_matrix[pool_of, torch.arange(len(wiring.readout), device=device)] = 1.0
        self.pool_matrix /= self.pool_matrix.sum(1, keepdim=True)

    def rates(self, role: str, drive: torch.Tensor, max_hz: float = 200.0) -> torch.Tensor:
        """drive: (batch, n_features) in [0, 1] -> input rates (n_input_neurons, batch)."""
        return self.maps[role] @ (drive.clamp(0, 1).T * max_hz)

    def pooled(self, readout_counts: torch.Tensor) -> torch.Tensor:
        """readout_counts: (n_readout, batch) -> (batch, n_pools) mean spikes per neuron."""
        return (self.pool_matrix @ readout_counts).T


def save_path(seed: int) -> str:
    return str(config.DATA_CACHE / f"wiring_seed{seed}.pt")
