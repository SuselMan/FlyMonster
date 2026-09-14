"""Run the physiology regression assays with the default overrides and print pass/fail."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flysim import assays, connectome  # noqa: E402
from flysim.physiology import DEFAULT, neuron_meta  # noqa: E402

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    con = connectome.load()
    metrics = assays.run(con, DEFAULT, neuron_meta(con))
    print(metrics)
    print(assays.check(metrics))
