"""Sensory transducers between the world and the connectome.

The connectome has no receptors. These turn physical quantities into firing
rates of real sensory neurons; the brain simulation then draws Poisson spikes
from the rates. Constants here are our modelling choices, not measurements.
"""
import numpy as np
import pandas as pd
import torch

# Synthetic odorants: which ORN types (glomeruli) each one activates and how
# strongly. Real flies have receptor tuning data (DoOR); this is a stand-in with
# the right structure: each odor drives a small, partly overlapping set.
ODORANTS = {
    "fruit": {"ORN_DM1": 1.0, "ORN_DM2": 0.8, "ORN_DM4": 0.7, "ORN_VA2": 0.5, "ORN_DL1": 0.3},
    "vinegar": {"ORN_DM1": 0.6, "ORN_DP1m": 1.0, "ORN_VM2": 0.7, "ORN_DC2": 0.5},
    "spider": {"ORN_DA2": 1.0, "ORN_VA3": 0.7, "ORN_DL3": 0.6, "ORN_VM4": 0.4},     # predators (spider, centipede)
    "fly": {"ORN_DA1": 1.0, "ORN_VA1d": 0.8, "ORN_VA1v": 0.6},   # pheromone-like
}


class Olfaction:
    """Concentration -> saturating receptor activation -> adaptation -> firing rate.

    r = r_max * relu(a - gamma * s) + r_spont,  a = C^n / (C^n + K^n) * affinity,
    ds/dt = (a - s) / tau_adapt.
    Left and right antenna are separate; in the fly each ORN population
    projects to both antennal lobes, so the brain decides what the difference means.
    """

    def __init__(self, meta: pd.DataFrame, batch: int, device, odorants: dict = ODORANTS,
                 K: float = 0.3, n: float = 1.5, r_max: float = 250.0, r_spont: float = 1.0,
                 gamma: float = 0.7, tau_adapt_ms: float = 1500.0):
        ct = meta.cell_type.fillna("").to_numpy()
        side = meta.side.fillna("").to_numpy()
        self.names = list(odorants)
        types = sorted({t for prof in odorants.values() for t in prof})
        self.K, self.n, self.r_max, self.r_spont = K, n, r_max, r_spont
        self.gamma, self.tau = gamma, tau_adapt_ms
        neuron_idx, unit_of = [], []          # unit = (type, side)
        units = [(t, s) for t in types for s in ("left", "right")]
        for u, (t, s) in enumerate(units):
            idx = np.flatnonzero((ct == t) & (side == s))
            neuron_idx.extend(idx.tolist())
            unit_of.extend([u] * len(idx))
        self.units = units
        self.input_idx = torch.tensor(neuron_idx, device=device)
        self.unit_of = torch.tensor(unit_of, device=device)
        aff = np.zeros((len(self.names), len(units)), dtype=np.float32)
        for o, name in enumerate(self.names):
            for t, w in odorants[name].items():
                aff[o, units.index((t, "left"))] = w
                aff[o, units.index((t, "right"))] = w
        self.affinity = torch.tensor(aff, device=device)                      # (odorants, units)
        self.side_of_unit = torch.tensor([0 if s == "left" else 1 for _, s in units], device=device)
        self.state = torch.zeros(batch, len(units), device=device)

    def keep(self, cols: torch.Tensor):
        self.state = self.state[cols]

    def add(self, n: int):
        self.state = torch.cat([self.state, torch.zeros(n, self.state.shape[1], device=self.state.device)])

    def rates(self, conc: torch.Tensor, dt_ms: float) -> torch.Tensor:
        """conc: (batch, odorants, 2) concentration at left/right antenna.

        Returns (n_orn_neurons, batch) rates in Hz; updates adaptation.
        """
        c = conc[:, :, self.side_of_unit]                                     # (B, odorants, units)
        sat = c.clamp(min=0) ** self.n / (c.clamp(min=0) ** self.n + self.K ** self.n)
        a = (sat * self.affinity).amax(1)                                     # strongest odorant per unit
        r = self.r_max * (a - self.gamma * self.state).clamp(min=0) + self.r_spont
        self.state += (a - self.state) * (dt_ms / self.tau)
        return r[:, self.unit_of].T
