"""Loading the FlyWire connectome (v783) as a sparse weight matrix."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from . import config


@dataclass
class Connectome:
    ids: np.ndarray          # FlyWire root id per model index
    weights: torch.Tensor    # sparse CSR (pre, post), signed synapse counts
    index_of: dict           # FlyWire root id -> model index

    @property
    def n(self) -> int:
        return len(self.ids)

    def indices(self, flywire_ids) -> list[int]:
        return [self.index_of[int(i)] for i in flywire_ids]


def load(use_cache: bool = True) -> Connectome:
    cache = config.DATA_CACHE / "connectome_783_pre_post.pt"
    if use_cache and cache.exists():
        d = torch.load(cache, weights_only=False)
        return Connectome(d["ids"], d["weights"], {int(i): k for k, i in enumerate(d["ids"])})

    ids = pd.read_csv(config.COMPLETENESS, index_col=0).index.to_numpy(dtype=np.int64)
    con = pd.read_parquet(
        config.CONNECTIVITY,
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    n = len(ids)
    # Rows are presynaptic: a spike of neuron i reads row i to find its targets.
    coo = torch.sparse_coo_tensor(
        torch.from_numpy(np.stack([con["Presynaptic_Index"].to_numpy(),
                                   con["Postsynaptic_Index"].to_numpy()])).long(),
        torch.from_numpy(con["Excitatory x Connectivity"].to_numpy().copy()).float(),
        size=(n, n),
        check_invariants=False,
    ).coalesce()
    weights = coo.to_sparse_csr()

    config.DATA_CACHE.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": ids, "weights": weights}, cache)
    return Connectome(ids, weights, {int(i): k for k, i in enumerate(ids)})


def annotations() -> pd.DataFrame:
    """FlyWire neuron annotations (cell class, type, side...), indexed by root id."""
    return pd.read_csv(config.ANNOTATIONS, sep="\t", index_col="root_id", low_memory=False)
