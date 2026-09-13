"""Paths and default parameters of the fly brain model.

Neuron and synapse parameters follow Shiu et al. 2024, Nature
("A Drosophila computational brain model reveals sensorimotor processing"),
https://github.com/philshiu/Drosophila_brain_model
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"

CONNECTIVITY = DATA_RAW / "Connectivity_783.parquet"
COMPLETENESS = DATA_RAW / "Completeness_783.csv"
ANNOTATIONS = DATA_RAW / "neuron_annotations.tsv"


@dataclass
class LIFParams:
    dt: float = 0.1          # ms, integration step
    v_0: float = -52.0       # mV, resting potential
    v_rst: float = -52.0     # mV, reset potential
    v_th: float = -45.0      # mV, spike threshold
    t_mbr: float = 20.0      # ms, membrane time constant
    tau: float = 5.0         # ms, synaptic time constant
    t_rfc: float = 2.2       # ms, refractory period
    t_dly: float = 1.8       # ms, synaptic delay
    w_syn: float = 0.275     # mV per synapse
    f_poi: float = 250.0     # Poisson input weight, in units of w_syn
    # Short-term synaptic depression (Tsodyks-Markram, depression only), per
    # presynaptic neuron: a spike transmits w * x, then x -= std_u * x, and x
    # recovers to 1 with std_tau. std_u = 0 disables it (original model).
    std_u: float = 0.0
    std_tau: float = 500.0   # ms
