"""Mushroom-body plasticity: dopamine-gated depression of Kenyon cell -> MBON synapses.

The fly's memory (Aso et al. 2014, Hige et al. 2015, Handler et al. 2019): a Kenyon cell spike
leaves a short eligibility trace in its output synapses; a dopamine neuron (DAN) spike arriving
in the same mushroom-body compartment while the trace is up depresses the KC -> MBON synapses of
that compartment. Reward DANs (PAM cluster) innervate the compartments of avoidance MBONs, so
a rewarded odor loses its power to drive avoidance; punishment DANs (PPL1) do the same to the
approach MBONs. Memories fade slowly.

What is data and what is ours:
- which synapses are plastic: every excitatory KC -> MBON synapse of the connectome;
- compartments: the DANs of an MBON are the DANs with direct synapses onto it (connectome);
- the rule and its constants (eligibility 1.5 s, dopamine window 0.3 s, rate, floor, forgetting)
  are ours; only depression is modelled (no potentiation, no consolidation).

Multipliers live in `brain.m` (one per plastic synapse per brain, FastBrain/FlyBrain
`enable_plasticity`); this class only updates them from spike counts of KCs and DANs.
"""
import math

import numpy as np
import torch

from . import connectome


class MushroomBody:
    def __init__(self, con: connectome.Connectome, meta, device, raw: connectome.Connectome | None = None,
                 eta: float = 0.02, tau_kc: float = 1.5, tau_dan: float = 0.3, tau_forget: float = 900.0,
                 m_min: float = 0.1, dan_min_syn: int = 5):
        """con: the connectome the brain runs (after physiology overrides; plastic synapse indices refer to
        its CSR). raw: the unmodified connectome for the compartment map, because the overrides drop the
        dopamine synapses (they are slow, not fast excitation) that mark which DANs innervate which MBON."""
        self.dev = torch.device(device)
        self.eta, self.tau_kc, self.tau_dan, self.tau_forget, self.m_min = eta, tau_kc, tau_dan, tau_forget, m_min
        cls = meta.cell_class.fillna("").to_numpy()
        ct = meta.cell_type.fillna("").to_numpy()
        w = con.weights
        crow, col, val = w.crow_indices(), w.col_indices(), w.values()
        pre = torch.repeat_interleave(torch.arange(con.n), crow[1:] - crow[:-1])
        is_kc, is_mbon, is_dan = (torch.from_numpy(cls == c) for c in ("Kenyon_Cell", "MBON", "DAN"))
        self.kc_idx = np.flatnonzero(cls == "Kenyon_Cell")
        self.mbon_idx = np.flatnonzero(cls == "MBON")
        self.dan_idx = np.flatnonzero(cls == "DAN")
        self.mbon_type = ct[self.mbon_idx]
        self.dan_type = ct[self.dan_idx]
        row_kc = torch.full((con.n,), -1, dtype=torch.long)
        row_kc[self.kc_idx] = torch.arange(len(self.kc_idx))
        row_mbon = torch.full((con.n,), -1, dtype=torch.long)
        row_mbon[self.mbon_idx] = torch.arange(len(self.mbon_idx))
        row_dan = torch.full((con.n,), -1, dtype=torch.long)
        row_dan[self.dan_idx] = torch.arange(len(self.dan_idx))
        # plastic synapses
        plastic = is_kc[pre] & is_mbon[col] & (val > 0)
        self.syn = torch.nonzero(plastic).squeeze(1)
        self.kc_row = row_kc[pre[self.syn]].to(self.dev)
        self.mbon_row = row_mbon[col[self.syn]].to(self.dev)
        # compartments: DAN -> MBON direct synapses of the raw connectome
        rw = (raw or con).weights
        rcrow, rcol, rval = rw.crow_indices(), rw.col_indices(), rw.values()
        rpre = torch.repeat_interleave(torch.arange(con.n), rcrow[1:] - rcrow[:-1])
        dm = is_dan[rpre] & is_mbon[rcol]
        D = torch.zeros(len(self.mbon_idx), len(self.dan_idx))
        D.index_put_((row_mbon[rcol[dm]], row_dan[rpre[dm]]), rval[dm].abs(), accumulate=True)
        D = (D >= dan_min_syn).float()
        self.D = (D / D.sum(1, keepdim=True).clamp(min=1)).to(self.dev)   # mean DAN trace of the compartment
        self.n_dan_of_mbon = D.sum(1).numpy()
        self.brain = None
        self.e = self.d = None

    def attach(self, brain):
        """Give the brain per-brain multipliers on the plastic synapses (before its graph is captured)."""
        brain.enable_plasticity(self.syn)
        self.brain = brain
        self.e = torch.zeros(len(self.kc_idx), brain.batch, device=self.dev)
        self.d = torch.zeros(len(self.dan_idx), brain.batch, device=self.dev)

    @torch.no_grad()
    def step(self, kc_counts: torch.Tensor, dan_counts: torch.Tensor, dt: float):
        """kc_counts (n_kc, batch), dan_counts (n_dan, batch): spikes in the last dt seconds."""
        self.e.mul_(math.exp(-dt / self.tau_kc)).add_(kc_counts)
        self.d.mul_(math.exp(-dt / self.tau_dan)).add_(dan_counts)
        dm = self.D @ self.d                                            # (n_mbon, batch)
        m = self.brain.m
        m.sub_(self.eta * dt * self.e[self.kc_row] * dm[self.mbon_row])
        m.add_((1.0 - m) * (dt / self.tau_forget)).clamp_(self.m_min, 1.0)

    def reset(self, cols):
        cols = torch.as_tensor(cols, device=self.dev, dtype=torch.long)
        self.e[:, cols] = 0
        self.d[:, cols] = 0
        self.brain.m[:, cols] = 1.0

    def summary(self, cols=None) -> dict:
        """Mean multiplier per MBON type (memory strength: 1 = nothing learned)."""
        m = self.brain.m if cols is None else self.brain.m[:, cols]
        per_syn = m.mean(1)
        out = {}
        for t in sorted(set(self.mbon_type)):
            sel = torch.from_numpy(np.isin(self.mbon_row.cpu().numpy(), np.flatnonzero(self.mbon_type == t))).to(self.dev)
            if sel.any():
                out[t] = float(per_syn[sel].mean())
        return out
