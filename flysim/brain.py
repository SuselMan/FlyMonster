"""Batched leaky integrate-and-fire simulation of the whole fly brain.

All brains in a batch share one weight matrix; only their state differs,
so many flies fit on one GPU.
"""
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
        self.delay_steps = round(self.p.t_dly / self.p.dt)
        self.refrac_steps = round(self.p.t_rfc / self.p.dt)
        self.reset()

    def reset(self):
        shape = (self.n, self.batch)
        dev = self.device
        self.v = torch.full(shape, self.p.v_0, device=dev)
        self.g = torch.zeros(shape, device=dev)
        self.refrac = torch.zeros(shape, dtype=torch.int16, device=dev)
        # Ring buffer of past spikes, to apply the synaptic delay.
        self.spike_buf = torch.zeros((self.delay_steps, *shape), device=dev)
        self.t = 0

    def step(self, poisson_rate: torch.Tensor | None = None) -> torch.Tensor:
        """Advance one dt. poisson_rate: Hz per neuron, shape (n,) or (n, batch).

        Returns a bool spike tensor of shape (n, batch).
        """
        p = self.p
        slot = self.t % self.delay_steps
        self._deliver(self.spike_buf[slot])

        if poisson_rate is not None:
            prob = (poisson_rate * p.dt * 1e-3).expand(self.n, self.batch)
            self.g += (torch.rand_like(self.g) < prob) * (p.f_poi * p.w_syn)

        active = self.refrac <= 0
        dv = (p.v_0 - self.v + self.g) * (p.dt / p.t_mbr)
        dg = -self.g * (p.dt / p.tau)
        self.v = torch.where(active, self.v + dv, self.v)
        self.g = torch.where(active, self.g + dg, self.g)

        spikes = self.v > p.v_th
        self.v = torch.where(spikes, torch.tensor(p.v_rst, device=self.device), self.v)
        self.g = torch.where(spikes, torch.zeros((), device=self.device), self.g)
        self.refrac = torch.where(spikes, torch.tensor(self.refrac_steps, dtype=torch.int16,
                                                       device=self.device), self.refrac - 1)

        self.spike_buf[slot] = spikes.float()
        self.t += 1
        return spikes

    def _deliver(self, arriving: torch.Tensor):
        """Add weights of spiking presynaptic neurons to their targets' g.

        Activity is sparse, so reading only the rows of neurons that spiked
        is much cheaper than a full sparse matrix product.
        """
        pre = arriving.any(1).nonzero().squeeze(1)
        if pre.numel() == 0:
            return
        starts = self.crow[pre]
        counts = self.crow[pre + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return
        offsets = torch.cumsum(counts, 0) - counts
        syn = torch.repeat_interleave(starts - offsets, counts) + torch.arange(total, device=self.device)
        src = torch.repeat_interleave(torch.arange(pre.numel(), device=self.device), counts)
        contrib = self.w[syn].unsqueeze(1) * arriving[pre][src]
        self.g.index_add_(0, self.post[syn], contrib)
