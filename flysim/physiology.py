"""Physiology overrides on top of "synapse count x transmitter sign".

Shiu et al. treat every synapse as fast and signed by the predicted
transmitter. Two known problems in that assumption matter for us:

- dopamine, serotonin and octopamine act mostly through slow metabotropic
  receptors, but the model counts them as fast excitation;
- in the antennal lobe, excitatory local neurons couple mainly electrically
  and weakly, while their many chemical synapses make the model's AL ignite
  into one global avalanche for any odor.

Overrides scale those synapses; everything else stays as in the connectome.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

from . import config, connectome
from .config import LIFParams

MODULATORS = ("dopamine", "serotonin", "octopamine")


@dataclass(frozen=True)
class Physiology:
    modulators_fast: float = 1.0   # scale of DA/5-HT/OA synapses as fast current (0 = off)
    al_exc_ln: float = 1.0         # scale of excitatory (positive-weight) antennal-lobe local neuron outputs
    std_u: float = 0.0             # short-term depression (see LIFParams)
    std_tau: float = 300.0

    def lif(self, dt: float = 0.5) -> LIFParams:
        return LIFParams(dt=dt, std_u=self.std_u, std_tau=self.std_tau)


ORIGINAL = Physiology()
# Chosen by scripts/physiology_scan.py: passes all assays in flysim/assays.py
# (odors separable, intensity coded, no runaway, sugar->MN9, looming->GF,
# vision->DNb05) while changing the fewest synapses. STD is off: even U=0.03
# weakened projection neuron -> Kenyon cell drive (68 vs 416 KCs for one odor).
DEFAULT = Physiology(modulators_fast=0.0, al_exc_ln=0.25, std_u=0.0)


def neuron_meta(con: connectome.Connectome) -> pd.DataFrame:
    """Annotations aligned to model indices (cached)."""
    cache = config.DATA_CACHE / "neuron_meta.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    a = connectome.annotations()
    a = a[~a.index.duplicated()].reindex(con.ids)
    cols = ["super_class", "cell_class", "cell_sub_class", "cell_type", "hemibrain_type", "top_nt", "side",
            "pos_x", "pos_y", "pos_z"]
    meta = a[cols].reset_index(drop=True)
    for c in cols:
        if meta[c].dtype == object:
            meta[c] = meta[c].astype("string")
    meta.to_parquet(cache)
    return meta


def apply(con: connectome.Connectome, phys: Physiology, meta: pd.DataFrame | None = None) -> connectome.Connectome:
    """Return a connectome with scaled synapses (rows = presynaptic neurons)."""
    if phys.modulators_fast == 1.0 and phys.al_exc_ln == 1.0:
        return con
    meta = neuron_meta(con) if meta is None else meta
    w = con.weights
    crow, col, val = w.crow_indices(), w.col_indices(), w.values().clone()
    counts = crow[1:] - crow[:-1]
    nt = meta.top_nt.fillna("").to_numpy()
    modulator = torch.repeat_interleave(torch.from_numpy(np.isin(nt, MODULATORS)), counts)
    val[modulator] *= phys.modulators_fast
    # The sign in the connectivity table does not always follow top_nt, so the
    # AL override acts on the actual excitatory weights of local neurons.
    ln = torch.repeat_interleave(torch.from_numpy((meta.cell_class.fillna("") == "ALLN").to_numpy()), counts)
    val[ln & (val > 0)] *= phys.al_exc_ln
    # drop synapses scaled to zero: identical dynamics, but spikes of e.g. octopamine neurons no longer
    # generate thousands of zero-weight events per spike
    keep = val != 0
    rows = torch.repeat_interleave(torch.arange(len(counts)), counts)
    kept = torch.bincount(rows[keep], minlength=len(counts))
    crow = torch.cat([torch.zeros(1, dtype=crow.dtype), torch.cumsum(kept, 0).to(crow.dtype)])
    return connectome.Connectome(con.ids, torch.sparse_csr_tensor(crow, col[keep], val[keep], w.shape), con.index_of)


def describe(phys: Physiology) -> dict:
    return asdict(phys)
