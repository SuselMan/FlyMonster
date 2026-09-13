"""Regression assays: known input -> output relations the model must keep.

All stimuli run in one batch. Each returns a metric; `check` turns metrics
into pass/fail against thresholds.
"""
import numpy as np
import pandas as pd
import torch

from . import connectome, neurons
from .brain import FlyBrain
from .physiology import Physiology, apply, neuron_meta

ODOR_TYPES = ("ORN_DA1", "ORN_VA1d", "ORN_VA1v", "ORN_DL3", "ORN_VL1", "ORN_VM4")

THRESHOLDS = {
    "odor_corr": ("<", 0.5),       # different glomeruli give different downstream patterns
    "odor_weak_strong": ("<", 0.6),  # 20 Hz odor gives clearly less than 150 Hz
    "runaway": ("<", 5.0),         # spikes/step 200-400 ms after any stimulus ends
    "mn9_sugar": (">", 20.0),      # sugar -> proboscis motor neuron MN9 (Hz)
    "mn9_bitter_ratio": ("<", 0.34),  # bitter drives MN9 much less than sugar
    "gf_looming": (">", 50.0),     # looming -> giant fiber (Hz)
    "dnb05_left": (">", 30.0),     # left visual input -> left DNb05 (Hz)
    "dnb05_lr_ratio": (">", 3.0),  # ... and much more than right DNb05
    "silent": ("<", 0.01),         # no input -> no spikes
}


def stimuli(con: connectome.Connectome, meta: pd.DataFrame) -> list[tuple[str, np.ndarray, float]]:
    ct = meta.cell_type.fillna("").to_numpy()
    sub = meta.cell_sub_class.fillna("").to_numpy()
    cls = meta.cell_class.fillna("").to_numpy()
    side = meta.side.fillna("").to_numpy()
    sup = meta.super_class.fillna("").to_numpy()
    out = [(f"odor {t}", np.flatnonzero(ct == t), 150.0) for t in ODOR_TYPES]
    out.append(("odor weak", np.flatnonzero(ct == ODOR_TYPES[0]), 20.0))
    out.append(("sugar", np.array(con.indices(i for i in neurons.SUGAR_GRN if i in con.index_of)), 150.0))
    out.append(("bitter", np.flatnonzero((cls == "gustatory") & (sub == "bitter")), 150.0))
    out.append(("looming", np.flatnonzero(np.isin(ct, ["LC4", "LPLC2"])), 150.0))
    vpn_left = np.flatnonzero((sup == "visual_projection") & (side == "left"))
    out.append(("vision left", vpn_left[_dn_reach_order(con, meta, vpn_left)][:60], 150.0))
    out.append(("nothing", np.array([], dtype=int), 0.0))
    return out


def _dn_reach_order(con, meta, idx):
    from .body import dn_reach
    dn = np.flatnonzero((meta.super_class == "descending").fillna(False).to_numpy())
    reach = dn_reach(con, dn)
    return np.argsort(-reach[idx], kind="stable")


@torch.no_grad()
def run(con: connectome.Connectome, phys: Physiology, meta: pd.DataFrame | None = None,
        stim_ms: float = 200.0, after_ms: float = 400.0) -> dict:
    meta = neuron_meta(con) if meta is None else meta
    model = apply(con, phys, meta)
    stims = stimuli(con, meta)
    B = len(stims)
    brain = FlyBrain(model, batch=B, params=phys.lif())
    idx = sorted({int(i) for _, g, _ in stims for i in g})
    row = {n: r for r, n in enumerate(idx)}
    rates = torch.zeros(len(idx), B, device=brain.device)
    for j, (_, g, hz) in enumerate(stims):
        if len(g):
            rates[[row[int(i)] for i in g], j] = hz
    idx_t = torch.tensor(idx, device=brain.device)
    dt = brain.p.dt
    on = torch.zeros(con.n, B, device=brain.device)
    for _ in range(int(stim_ms / dt)):
        on += brain.step(idx_t, rates)
    for _ in range(int(after_ms / 2 / dt)):
        brain.step()
    late = torch.zeros(B, device=brain.device)
    for _ in range(int(after_ms / 2 / dt)):
        late += brain.step().sum(0)
    hz = on.cpu().numpy() / (stim_ms / 1000)
    late = late.cpu().numpy() / int(after_ms / 2 / dt)
    names = [s[0] for s in stims]
    col = {n: i for i, n in enumerate(names)}

    ct = meta.cell_type.fillna("").to_numpy()
    side = meta.side.fillna("").to_numpy()
    sup = meta.super_class.fillna("").to_numpy()
    cls = meta.cell_class.fillna("").to_numpy()
    readout = np.flatnonzero(np.isin(sup, ["descending", "motor"]) | np.isin(cls, ["LHLN", "MBON", "ALPN"]))
    odors = np.stack([hz[readout, col[f"odor {t}"]] for t in ODOR_TYPES])
    with np.errstate(invalid="ignore"):
        c = np.corrcoef(odors)
    offdiag = c[~np.eye(len(ODOR_TYPES), dtype=bool)]
    mn9 = con.index_of[neurons.MN9]
    gf = np.flatnonzero(ct == "DNp01")
    b05l = np.flatnonzero((ct == "DNb05") & (side == "left"))
    b05r = np.flatnonzero((ct == "DNb05") & (side == "right"))
    strong = hz[readout, col[f"odor {ODOR_TYPES[0]}"]].sum()
    return {
        "odor_corr": float(np.nanmean(offdiag)) if np.isfinite(offdiag).any() else 1.0,
        "odor_weak_strong": float(hz[readout, col["odor weak"]].sum() / max(strong, 1e-9)),
        "odor_downstream_hz": float(strong),
        "runaway": float(late[:-1].max()),
        "mn9_sugar": float(hz[mn9, col["sugar"]]),
        "mn9_bitter_ratio": float(hz[mn9, col["bitter"]] / max(hz[mn9, col["sugar"]], 1e-9)),
        "gf_looming": float(hz[gf, col["looming"]].mean()),
        "dnb05_left": float(hz[b05l, col["vision left"]].mean()),
        "dnb05_lr_ratio": float(hz[b05l, col["vision left"]].mean() / max(hz[b05r, col["vision left"]].mean(), 1.0)),
        "silent": float(hz[:, col["nothing"]].sum()),
    }


def check(metrics: dict) -> dict:
    out = {}
    for k, (op, thr) in THRESHOLDS.items():
        v = metrics[k]
        out[k] = v < thr if op == "<" else v > thr
    return out
