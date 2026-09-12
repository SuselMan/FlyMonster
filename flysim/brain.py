"""Batched leaky integrate-and-fire simulation of the whole fly brain.

All brains in a batch share one weight matrix; only their state differs,
so many flies fit on one GPU. Activity is sparse, so synaptic input is
delivered only from neurons that actually spiked.
"""
import math

import torch

from .config import LIFParams
from .connectome import Connectome


class FlyBrain:
    def __init__(self, connectome: Connectome, batch: int = 1,
                 params: LIFParams | None = None, device: str | None = None):
        self.p = params or LIFParams()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch = batch
        self.n = connectome.n
        w = connectome.weights
        self.crow = w.crow_indices().to(self.device)
        self.post = w.col_indices().to(self.device)
        self.w = (w.values() * self.p.w_syn).to(self.device)
        self.delay_steps = max(1, round(self.p.t_dly / self.p.dt))
        self.refrac_steps = max(1, round(self.p.t_rfc / self.p.dt))
        self.g_decay = math.exp(-self.p.dt / self.p.tau)
        self.reset()

    def reset(self, which: torch.Tensor | None = None):
        """Reset all brains, or only the batch columns in `which` (bool mask)."""
        shape = (self.n, self.batch)
        dev = self.device
        if which is None:
            self.v = torch.full(shape, self.p.v_0, device=dev)
            self.g = torch.zeros(shape, device=dev)
            self.refrac = torch.zeros(shape, dtype=torch.int16, device=dev)
            # Ring buffer of past spikes, to apply the synaptic delay.
            self.spike_buf = torch.zeros((self.delay_steps, *shape), dtype=torch.bool, device=dev)
            self.t = 0
            return
        self.v[:, which] = self.p.v_0
        self.g[:, which] = 0
        self.refrac[:, which] = 0
        self.spike_buf[:, :, which] = False

    def step(self, input_idx: torch.Tensor | None = None,
             input_rate: torch.Tensor | None = None) -> torch.Tensor:
        """Advance one dt.

        input_idx:  (k,) neuron indices receiving Poisson input
        input_rate: (k, batch) rates in Hz
        Returns a bool spike tensor of shape (n, batch).
        """
        p = self.p
        slot = self.t % self.delay_steps
        self._deliver(self.spike_buf[slot])

        if input_idx is not None:
            hits = torch.rand_like(input_rate) < input_rate * (p.dt * 1e-3)
            self.g.index_add_(0, input_idx, hits * (p.f_poi * p.w_syn))

        active = self.refrac <= 0
        self.v = torch.where(active, self.v + (p.v_0 - self.v + self.g) * (p.dt / p.t_mbr), self.v)
        self.g = torch.where(active, self.g * self.g_decay, self.g)

        spikes = self.v > p.v_th
        self.v.masked_fill_(spikes, p.v_rst)
        self.g.masked_fill_(spikes, 0.0)
        self.refrac.sub_(1).clamp_(min=0).masked_fill_(spikes, self.refrac_steps)

        self.spike_buf[slot] = spikes
        self.t += 1
        return spikes

    def _deliver(self, arriving: torch.Tensor):
        """Add weights of spiking presynaptic neurons to their targets' g.

        Works on individual (neuron, brain) spikes, so cost is proportional
        to the number of synaptic events, not to batch size.
        """
        pre, col = arriving.nonzero(as_tuple=True)
        if pre.numel() == 0:
            return
        starts = self.crow[pre]
        counts = self.crow[pre + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return
        offsets = torch.cumsum(counts, 0) - counts
        syn = torch.repeat_interleave(starts - offsets, counts) + torch.arange(total, device=self.device)
        flat = self.post[syn] * self.batch + torch.repeat_interleave(col, counts)
        self.g.view(-1).index_add_(0, flat, self.w[syn])
