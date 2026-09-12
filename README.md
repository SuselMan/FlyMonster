# Fly team

Two simulated fruit fly brains (FlyWire connectome) that must cooperate:
the "Eye" fly sees objects far away, the "Nose" fly smells whether they are food.
The goal is to let them evolve a communication channel and decode it.

## Layout

```
flysim/            Python package: simulation
  config.py        paths and LIF parameters (Shiu et al. 2024)
  connectome.py    FlyWire v783 -> sparse weight matrix
  brain.py         batched whole-brain LIF simulation (PyTorch, CPU/CUDA)
  neurons.py       named neuron groups (sugar GRNs, MN9, ...)
  world.py         seeded maze: loops, traps, smell and vision
  body.py          which neurons get senses, which are read out
  team.py          maze episodes with brains in the loop, curriculum levels
scripts/
  download_data.py fetch connectome and annotations into data/raw
  smoke_test.py    run the brain, check the sugar -> MN9 response, measure speed
  probe_senses.py  does every sense reach descending neurons without runaway activity
  show_maps.py     print generated mazes
  evolve.py        evolution of Eye + Nose pairs, logs and replays in results/<run>
  serve.py         web viewer: python scripts/serve.py, open http://<pc>:8080
viewer/            viewer page
setup/             one-click setup of the GPU PC
data/              downloaded data and caches (not in git)
results/           run outputs (not in git)
index.js           Node side, later: web UI for watching the flies
```

## Setup

```
python -m venv .venv
.venv\Scripts\pip install numpy pandas pyarrow
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu    # or cu126 on the GPU PC
.venv\Scripts\python scripts\download_data.py
.venv\Scripts\python scripts\smoke_test.py --ms 100
```

## Data

- Connectome: FlyWire v783, as prepared in
  [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)
- Annotations: [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)
