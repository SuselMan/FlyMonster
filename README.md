# FlyMonster

Experiments with the whole-brain fruit fly model (FlyWire v783 connectome,
leaky integrate-and-fire model of Shiu et al. 2024), running on a GPU:
a maze with talking flies (failed, kept for reference), Flappy Fly,
an interactive brain explorer, and a 2D world where flies with full brains live.

## Viewer

`python scripts/serve.py`, then open `http://<pc>:8080`:

- `/viewer/life.html` — flies living in an arena, 1:1 playback of the simulation
- `/viewer/brain.html` — stimulate a sense, watch activity spread through all neurons
- `/viewer/flappy.html` — the fly brain playing Flappy Bird
- `/` — maze experiment

## Layout

```
flysim/
  config.py        paths, LIF parameters (+ optional short-term depression)
  connectome.py    FlyWire v783 -> sparse weight matrix
  brain.py         batched whole-brain LIF simulation (PyTorch, optional CuPy kernel)
  physiology.py    overrides of "synapse count x transmitter sign" (see below)
  assays.py        regression assays the model must pass
  senses.py        sensory transducers (olfaction: saturation, adaptation)
  explorer.py      backend of the brain explorer
  flappy*.py       Flappy Fly
  life/            the arena world: arena.py (food ecology, flowers, spider webs, centipedes, ants,
                   birds, day/night, seasons, temperature), sim.py (flies)
  world.py body.py team.py   maze experiment
scripts/
  physiology_scan.py  scan overrides against assays
  mb_causal_scan.py   does KC->MBON plasticity change descending neurons?
  life_run.py         run and record the arena
  serve.py            web viewer
  ...
viewer/            pages
setup/             one-click setup of the GPU PC
```

## What is data and what is ours

Between sensory neurons and descending/motor neurons everything is the
connectome and the published LIF model. Around it:

- **Physiology overrides** (`physiology.py`, default chosen by `physiology_scan.py`):
  dopamine/serotonin/octopamine synapses are not treated as fast excitation
  (they act through slow receptors); excitatory outputs of antennal-lobe local
  neurons are scaled x0.25. Without this any odor ignites one self-sustaining
  avalanche and all odors look the same. With it: odors separable
  (corr 0.02), intensity coded, no runaway, and sugar->MN9, looming->giant
  fiber, vision->DNb05 are unchanged.
- **Photoreceptors are not used**: histamine is missing from FlyWire's
  transmitter predictions, so the first visual synapse has the wrong sign.
  Vision enters at visual projection neurons.
- **Sensory transducers** (odor saturation/adaptation, vision salience,
  looming from bird shadows) are modelling choices.
- **Body**: walking is an innate generator (leg circuits are in the ventral
  nerve cord, not in FAFB). The brain steers with DNa01/DNa02, walks backward
  with MDN, takes off with the giant fiber DNp01, feeds with MN9. The body
  adapts to a sustained left-right DNa difference (the single DNa neurons of
  this connectome carry a static left bias).
- **Hunger** lowers the threshold of NPF neurons and raises sugar sensitivity.
- **One female fly's wiring**: every fly has the same connectome.
- **The world is scripted** (`life/arena.py`), the flies are not: nothing in
  the world tells a fly where to go. Food has causes and is finite: apples only
  when a viewer drops them, predator droppings after a meal, bodies of flies
  that died of hunger/age after decomposing. Droppings and bodies reuse the
  "vinegar" odorant (fermenting matter) rather than an invented new receptor
  profile. Food size, odor and taste follow the remaining amount; flies and
  ants eat it away. The world starts with a few old droppings. Two apple
  trees drop apples under their crowns from midsummer to mid-autumn (world
  rule, so the population can live long enough to reproduce).
- **Animals**: spiders build webs where a running heatmap says flies walk
  (or near food) and collect flies from their own webs; webs weaken with age
  and when flies tear free; centipedes hunt by sight/vibration. The world
  starts with one spider; the viewer can release more spiders and centipedes
  (up to 4 of each) like it drops apples; both have hunger (hunt only when hungry, slow down
  and rest when starving); ants forage and carry food home; birds dive in
  daylight. Flies perceive all of them only through vision salience, looming,
  odor and touch. Tuning numbers are ours (ArenaConfig).
- **Centipedes and flowers**: centipedes age and starve to death, their large
  bodies decompose into food, newcomers walk in from the map edge. Flowers hold
  refilling nectar (the "fruit" odorant at lower strength). Pollen is fly body
  state: feeding on a different flower pollinates it and may sprout a seedling.
- **Seasons**: a 75-minute year on top of day/night. Temperature = annual +
  daily cycle + analytic microclimate (leaf-litter shelters and stones warmer
  in the cold, open ground colder). Cold slows decomposition (Q10 = 2), stops
  flowers, makes predators and ants hibernate, reduces birds. Flies do not
  react to temperature yet: `Arena.temperature()` / `Life.fly_temperature()`
  are the hooks, and each fly row in the frame carries its local temperature.
- **Flight metrics** (long flights, landing across water) are counted for
  checking, they do not trigger anything.

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
- Reference for biological assumptions: [vaibhavkedarisetti/fruit-fly-lab](https://github.com/vaibhavkedarisetti/fruit-fly-lab)
