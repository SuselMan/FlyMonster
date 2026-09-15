"""Whole-brain LIF simulation replayed as a CUDA graph.

Same equations as brain.FlyBrain (without short-term depression), but every
tensor has a fixed shape and nothing waits for the CPU, so a block of steps
can be captured once and replayed with a single call:

- spike delivery uses fixed-size buffers (at most `max_spikes` spiking
  neurons and `max_events` synaptic events per step, across the batch);
  an overflow flag is kept on the GPU and checked once per block;
- the batch size is fixed; unused columns simply receive no input;
- optional per-brain plasticity: a chosen set of synapses carries a multiplier per brain
  (`enable_plasticity`), read by both delivery kernels; the multipliers live in `self.m` and are
  changed in place outside the graph (flysim/plasticity.py).

Replaying removes the per-operation launch and synchronisation overhead,
which dominates when activity is sparse.
"""
import math

import torch

from .brain import _LIF_KERNEL as _LIF
from .config import LIFParams
from .connectome import Connectome

if _LIF is not None:
    import cupy

    # Spike delivery, step 1: one short GPU thread per synaptic event. Event e
    # finds its spike j by binary search in the cumulative synapse counts and
    # writes (flat target index, weight). Step 2 (torch index_add_) adds the
    # buffer into g, which is race-free.
    _FILL = cupy.RawKernel(r"""
    extern "C" __global__ void fill(const long long* pre, const long long* col, const long long* off,
                                    const long long* crow, const long long* post, const float* w,
                                    const int* pmap, const float* m,
                                    long long* tgt, float* wout, const int k, const int batch, const long long emax) {
        long long e = (long long)blockDim.x * blockIdx.x + threadIdx.x;
        if (e >= emax) return;
        if (e >= off[k - 1]) { wout[e] = 0; tgt[e] = 0; return; }
        int lo = 0, hi = k - 1;                       // first j with off[j] > e
        while (lo < hi) { int mid = (lo + hi) / 2; if (off[mid] > e) hi = mid; else lo = mid + 1; }
        long long before = lo > 0 ? off[lo - 1] : 0;
        long long s = crow[pre[lo]] + (e - before);
        tgt[e] = post[s] * batch + col[lo];
        int p = pmap[s];
        wout[e] = p >= 0 ? w[s] * m[(long long)p * batch + col[lo]] : w[s];
    }
    """, "fill")

    # EXPERIMENTAL, off by default: measured ~200 ns per synaptic event in a real life run (GPU threads
    # are slow at long serial loops), i.e. ~20x slower than the event-buffer path. Kept for reference.
    # Spike delivery in one pass, work proportional to the actual events: one GPU thread per
    # batch column walks that column's spiking neurons and adds each synapse weight into g.
    # Columns never write the same element (index post * batch + col), so there is no race,
    # and there is no fixed-size event buffer to scan every step.
    _DELIVER = cupy.RawKernel(r"""
    extern "C" __global__ void deliver(const long long* pre, const long long* colstart, const long long* crow,
                                       const long long* post, const float* w, const int* pmap, const float* m,
                                       float* g, const int batch, const long long kmax) {
        int c = blockDim.x * blockIdx.x + threadIdx.x;
        if (c >= batch) return;
        long long k0 = colstart[c], k1 = colstart[c + 1];
        if (k1 > kmax) k1 = kmax;
        for (long long k = k0; k < k1; k++) {
            long long i = pre[k];
            long long s1 = crow[i + 1];
            for (long long s = crow[i]; s < s1; s++) {
                int p = pmap[s];
                g[post[s] * batch + c] += p >= 0 ? w[s] * m[(long long)p * batch + c] : w[s];
            }
        }
    }
    """, "deliver")
else:
    _FILL = None
    _DELIVER = None


class FastBrain:
    def __init__(self, connectome: Connectome, batch: int, params: LIFParams, input_idx: torch.Tensor,
                 read_idx: torch.Tensor, bias_idx: torch.Tensor, steps: int = 20,
                 max_spikes: int = 4096, max_events: int = 400_000, device: str = "cuda", use_kernel: bool = True,
                 deliver: bool = False):
        assert params.std_u == 0, "short-term depression is not supported in the graph path"
        self.p = p = params
        self.dev = torch.device(device)
        self.n, self.batch, self.steps = connectome.n, batch, steps
        self.delay_steps = max(1, round(p.t_dly / p.dt))
        assert steps % self.delay_steps == 0, "block length must be a multiple of the synaptic delay"
        self.refrac_steps = max(1, round(p.t_rfc / p.dt))
        self.k_mbr = p.dt / p.t_mbr
        self.g_decay = math.exp(-p.dt / p.tau)
        self.max_spikes, self.max_events = max_spikes, max_events
        w = connectome.weights
        crow = w.crow_indices().to(self.dev)
        self.crow = torch.cat([crow, crow[-1:]])            # extra row: padding "neuron" n has no synapses
        self.post = w.col_indices().to(self.dev)
        self.w = (w.values() * p.w_syn).to(self.dev)
        self.nnz = self.w.numel()
        # plasticity: pmap[s] = row of the multiplier table for synapse s (-1: fixed weight)
        self.pmap = torch.full((self.nnz,), -1, dtype=torch.int32, device=self.dev)
        self.m = torch.ones(1, batch, device=self.dev)
        self.input_idx, self.read_idx, self.bias_idx = input_idx.to(self.dev), read_idx.to(self.dev), bias_idx.to(self.dev)
        shape = (self.n, batch)
        self.v = torch.full(shape, p.v_0, device=self.dev)
        self.g = torch.zeros(shape, device=self.dev)
        self.refrac = torch.zeros(shape, dtype=torch.int16, device=self.dev)
        self.spike_buf = torch.zeros((self.delay_steps, *shape), dtype=torch.bool, device=self.dev)
        self.events = torch.arange(max_events, device=self.dev)
        # static inputs and outputs of the graph
        self.rates = torch.zeros(len(self.input_idx), batch, device=self.dev)
        self.bias = torch.zeros(batch, device=self.dev)          # added to g of bias neurons each step (mV)
        self.counts = torch.zeros(len(self.read_idx), batch, device=self.dev)
        self.overflow = torch.zeros((), device=self.dev)
        self.overflow_kind = torch.zeros(2, device=self.dev)       # [event buffer, spike list] overflowed
        self._zero1 = torch.zeros(1, dtype=torch.long, device=self.dev)
        self._zero_f = torch.zeros((), device=self.dev)
        self.graph = None
        self.use_kernel = _LIF is not None and use_kernel
        self.deliver = self.use_kernel and deliver          # one-pass delivery (no event buffer)
        if self.use_kernel:
            self._cp = cupy.from_dlpack
            self._cp_v, self._cp_g = cupy.from_dlpack(self.v), cupy.from_dlpack(self.g)
            self._cp_gflat = cupy.from_dlpack(self.g.view(-1))
            self._cp_r = cupy.from_dlpack(self.refrac)
            self._cp_s = [cupy.from_dlpack(self.spike_buf[s]) for s in range(self.delay_steps)]
            self._cp_crow, self._cp_post = cupy.from_dlpack(self.crow), cupy.from_dlpack(self.post)
            self._cp_w = cupy.from_dlpack(self.w)
            self._cp_pmap, self._cp_m = cupy.from_dlpack(self.pmap), cupy.from_dlpack(self.m)
            self._alloc_events()

    def enable_plasticity(self, syn: torch.Tensor):
        """Give synapses `syn` (indices into the CSR values) a multiplier per brain: self.m[k, b] for
        syn[k] in brain b, initially 1. Call before the first run (the graph reads these buffers)."""
        assert self.graph is None, "enable plasticity before the graph is captured"
        syn = syn.to(self.dev, torch.long)
        self.pmap.fill_(-1)
        self.pmap[syn] = torch.arange(len(syn), dtype=torch.int32, device=self.dev)
        self.m = torch.ones(len(syn), self.batch, device=self.dev)
        if self.use_kernel:
            self._cp_pmap, self._cp_m = cupy.from_dlpack(self.pmap), cupy.from_dlpack(self.m)

    def _alloc_events(self):
        self.events = torch.arange(self.max_events, device=self.dev)
        if self.use_kernel:
            self.ev_tgt = torch.zeros(self.max_events, dtype=torch.long, device=self.dev)
            self.ev_w = torch.zeros(self.max_events, device=self.dev)
            self._cp_tgt, self._cp_wout = cupy.from_dlpack(self.ev_tgt), cupy.from_dlpack(self.ev_w)

    # --- one step, fixed shapes, no host synchronisation ---------------------
    def _step(self, slot: int):
        p, B, n = self.p, self.batch, self.n
        if self.deliver:
            # spikes listed column by column (flat index col * n + neuron) with per-column offsets
            arriving = self.spike_buf[slot].t().reshape(-1)
            flat = torch.nonzero_static(arriving, size=self.max_spikes, fill_value=n * B).squeeze(1)
            pre = flat % n
            colstart = torch.cat([self._zero1, torch.cumsum(self.spike_buf[slot].sum(0), 0)])
            _DELIVER(((B + 63) // 64,), (64,), (self._cp(pre), self._cp(colstart), self._cp_crow, self._cp_post,
                                               self._cp_w, self._cp_pmap, self._cp_m, self._cp_gflat, cupy.int32(B),
                                               cupy.int64(self.max_spikes)))
            sp_over = (colstart[-1] > self.max_spikes).float()
            self.overflow_kind.copy_(torch.maximum(self.overflow_kind, torch.stack([self._zero_f, sp_over])))
            self.overflow.copy_(torch.maximum(self.overflow, sp_over))
            return self._integrate(slot)
        if self.use_kernel:
            arriving = self.spike_buf[slot].reshape(-1)
            flat = torch.nonzero_static(arriving, size=self.max_spikes, fill_value=n * B).squeeze(1)
            pre, col = flat // B, flat % B
            cnt = self.crow[pre + 1] - self.crow[pre]
            off = torch.cumsum(cnt, 0)
            K, E = self.max_spikes, self.max_events
            _FILL(((E + 1023) // 1024,), (1024,), (self._cp(pre), self._cp(col), self._cp(off), self._cp_crow,
                                                    self._cp_post, self._cp_w, self._cp_pmap, self._cp_m,
                                                    self._cp_tgt, self._cp_wout, cupy.int32(K), cupy.int32(B), cupy.int64(E)))
            self.g.view(-1).index_add_(0, self.ev_tgt, self.ev_w)
            # overflow: more events than the buffer, or the spike list is full (its last entry is real)
            ev_over, sp_over = (off[-1] > E).float(), (flat[-1] < n * B).float()
            self.overflow_kind.copy_(torch.maximum(self.overflow_kind, torch.stack([ev_over, sp_over])))
            self.overflow.copy_(torch.maximum(self.overflow, torch.maximum(ev_over, sp_over)))
            return self._integrate(slot)
        arriving = self.spike_buf[slot].reshape(-1)
        flat = torch.nonzero_static(arriving, size=self.max_spikes, fill_value=n * B).squeeze(1)
        pre, col = flat // B, flat % B
        start = self.crow[pre]
        cnt = self.crow[pre + 1] - start
        off = torch.cumsum(cnt, 0)
        total = off[-1]
        j = torch.searchsorted(off, self.events, right=True).clamp(max=self.max_spikes - 1)
        syn = (start[j] + self.events - (off[j] - cnt[j])).clamp(max=self.nnz - 1)
        valid = self.events < total
        wv = self.w[syn] * valid
        pj = self.pmap[syn].long()
        wv = torch.where(pj >= 0, wv * self.m[pj.clamp(min=0), col[j]], wv)
        self.g.view(-1).index_add_(0, self.post[syn] * B + col[j], wv)
        ev_over, sp_over = (total > self.max_events).float(), (arriving.sum() > self.max_spikes).float()
        self.overflow_kind.copy_(torch.maximum(self.overflow_kind, torch.stack([ev_over, sp_over])))
        self.overflow.copy_(torch.maximum(self.overflow, torch.maximum(ev_over, sp_over)))
        return self._integrate(slot)

    def _integrate(self, slot: int):
        p = self.p
        hits = torch.rand_like(self.rates) < self.rates * (p.dt * 1e-3)
        self.g.index_add_(0, self.input_idx, hits * (p.f_poi * p.w_syn))
        if len(self.bias_idx):
            self.g[self.bias_idx] += self.bias * (p.dt / p.tau)

        spikes = self.spike_buf[slot]
        if self.use_kernel:
            # the fused CuPy kernel, launched on torch's current stream so the graph records it
            _LIF(p.v_0, self.k_mbr, self.g_decay, p.v_th, p.v_rst, self.refrac_steps,
                 self._cp_v, self._cp_g, self._cp_r, self._cp_s[slot])
            return spikes
        active = self.refrac <= 0
        self.v.add_((p.v_0 - self.v + self.g) * self.k_mbr * active)
        self.g.mul_(torch.where(active, self.g_decay, 1.0))
        new = self.v > p.v_th
        self.v.masked_fill_(new, p.v_rst)
        self.g.masked_fill_(new, 0.0)
        self.refrac.sub_(1).clamp_(min=0).masked_fill_(new, self.refrac_steps)
        spikes.copy_(new)
        return spikes

    def _block(self):
        self.counts.zero_()
        stream = cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream) if self.use_kernel else None
        if stream is not None:
            stream.use()
        for s in range(self.steps):
            spikes = self._step(s % self.delay_steps)
            self.counts += spikes[self.read_idx]

    def capture(self):
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(3):
                self._block()
        torch.cuda.current_stream().wait_stream(side)
        self.reset_all()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._block()
        self.reset_all()

    def run_eager(self):
        """Same block without the graph (for timing comparisons)."""
        self._block()

    def run(self):
        """Advance `steps` steps; spike counts of readout neurons land in self.counts."""
        if self.graph is None:
            self.capture()
        self.graph.replay()

    # --- state management (outside the graph, in place) ----------------------
    def reset_all(self):
        self.m.fill_(1.0)
        self.v.fill_(self.p.v_0)
        self.g.zero_()
        self.refrac.zero_()
        self.spike_buf.zero_()
        self.overflow.zero_()
        self.overflow_kind.zero_()

    def reset_columns(self, cols):
        cols = torch.as_tensor(cols, device=self.dev, dtype=torch.long)
        self.v[:, cols] = self.p.v_0
        self.g[:, cols] = 0
        self.m[:, cols] = 1.0                       # a fresh brain has no memories
        self.refrac[:, cols] = 0
        self.spike_buf[:, :, cols] = False
