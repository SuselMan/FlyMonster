"""Timing of the life world in chunks of 50 steps, with buffer regrowth counts (delivery kernel debug)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402

from flysim.life.sim import Life, LifeConfig  # noqa: E402

deliver = "--no-deliver" not in sys.argv
life = Life(LifeConfig(n_flies=7, max_flies=9, seed=3))
b = life.brain
b.deliver = deliver and b.deliver
grows = {"n": 0}
orig = life._check_overflow


def wrapped(*a, **k):
    grows["n"] += 1
    return orig(*a, **k)


life._check_overflow = wrapped
for chunk in range(6):
    t0 = time.perf_counter()
    for _ in range(50):
        life.step()
    torch.cuda.synchronize()
    print(f"50 steps {time.perf_counter() - t0:6.2f}s  deliver={b.deliver} spike buffer {b.max_spikes} "
          f"regrows {grows['n']} alive {len(life.ids)}", flush=True)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    b.run()
torch.cuda.synchronize()
print(f"block alone {(time.perf_counter() - t0) / 20 * 1000:.2f} ms", flush=True)
