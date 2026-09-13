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


class Wind:
    """Wind on the antennae -> Johnston's organ wind/gravity neurons (JO-C, JO-E).

    Each antenna (arista) is deflected by the wind component along its axis; the
    antennae point ~45 deg to each side of the head: d_L = s * cos(rel - 45 deg),
    d_R = s * cos(rel + 45 deg), rel = direction the wind comes FROM, relative to
    heading (+ = left). JO-C neurons fire for one deflection direction, JO-E for the
    other (Kamikouchi et al. 2009, Yorozu et al. 2009). Which group is "push" and the
    gains are our modelling choice.
    """

    def __init__(self, meta: pd.DataFrame, device, r_max: float = 150.0, full_speed: float = 0.8):
        ct = meta.cell_type.fillna("").to_numpy()
        side = meta.side.fillna("").to_numpy()
        sub = meta.cell_sub_class.fillna("").to_numpy()
        wind = sub == "wind_gravity"
        groups = [(np.char.startswith(ct.astype(str), "JO-C") & wind, s, +1) for s in ("left", "right")] + \
                 [(np.char.startswith(ct.astype(str), "JO-E") & wind, s, -1) for s in ("left", "right")]
        idx, antenna, sign = [], [], []
        for mask, s, sg in groups:
            sel = np.flatnonzero(mask & (side == s))
            idx.extend(sel.tolist())
            antenna.extend([0 if s == "left" else 1] * len(sel))
            sign.extend([sg] * len(sel))
        self.input_idx = torch.tensor(idx, device=device)
        self.antenna = torch.tensor(antenna, device=device)
        self.sign = torch.tensor(sign, dtype=torch.float32, device=device)
        self.r_max, self.full_speed = r_max, full_speed

    def rates(self, speed: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        """speed, rel: (batch,) -> (n_neurons, batch) Hz."""
        s = (speed / self.full_speed).clamp(0, 1)
        d = torch.stack([s * torch.cos(rel - np.pi / 4), s * torch.cos(rel + np.pi / 4)])   # (2, batch)
        return self.r_max * (d[self.antenna] * self.sign[:, None]).clamp(min=0)


class Compass:
    """Head direction as a bump of activity on the E-PG ring of the central complex.

    FlyWire does not annotate E-PG wedges; the ring order is recovered from the
    connectome itself (spectral embedding of each E-PG's input/output partner
    profile gives a degenerate pair of eigenvectors = a circle). In the fly the
    bump is set by ring neurons from the visual scene; here we inject it directly
    (modelling choice: the compass works and is anchored to the world).
    """

    def __init__(self, con, meta: pd.DataFrame, device, r_max: float = 80.0, kappa: float = 2.5):
        ct = meta.cell_type.fillna("").to_numpy()
        idx = np.flatnonzero(ct == "EPG")
        self.phase = torch.tensor(ring_phase(con, idx), dtype=torch.float32, device=device)
        self.input_idx = torch.tensor(idx, device=device)
        self.r_max, self.kappa = r_max, kappa

    def rates(self, heading: torch.Tensor) -> torch.Tensor:
        """heading: (batch,) rad -> (n_epg, batch) Hz."""
        return self.r_max * torch.exp(self.kappa * (torch.cos(self.phase[:, None] - heading[None]) - 1))


def ring_phase(con, idx: np.ndarray) -> np.ndarray:
    """Angle of each neuron on the ring recovered from its partner profile (normalized spectral embedding)."""
    w = con.weights
    crow, col, val = w.crow_indices().cpu().numpy(), w.col_indices().cpu().numpy(), w.values().cpu().numpy()
    pre = np.repeat(np.arange(len(crow) - 1), np.diff(crow))
    pos = np.full(len(crow) - 1, -1)
    pos[idx] = np.arange(len(idx))
    om, im = pos[pre] >= 0, pos[col] >= 0
    parts = np.unique(np.concatenate([col[om], pre[im]]))
    pp = np.full(len(crow) - 1, -1)
    pp[parts] = np.arange(len(parts))
    P = np.zeros((len(idx), 2 * len(parts)))
    np.add.at(P, (pos[pre[om]], pp[col[om]]), np.abs(val[om]))
    np.add.at(P, (pos[col[im]], len(parts) + pp[pre[im]]), np.abs(val[im]))
    P = np.sqrt(P)
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-9
    S = P @ P.T
    d = 1 / np.sqrt(S.sum(1))
    _, vec = np.linalg.eigh(d[:, None] * S * d[None])
    return np.arctan2(vec[:, -3], vec[:, -2])
