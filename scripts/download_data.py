"""Download the FlyWire v783 connectome (Shiu et al. 2024) and neuron annotations."""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flysim import config  # noqa: E402

SHIU = ("https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/"
        "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960")
ANN = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
       "8587524c1748ce5ef2080822a2fc890fc03bf597/supplemental_files")

FILES = {
    config.COMPLETENESS: (f"{SHIU}/Completeness_783.csv", 3327347),
    config.CONNECTIVITY: (f"{SHIU}/Connectivity_783.parquet", 100804642),
    config.ANNOTATIONS: (f"{ANN}/Supplemental_file1_neuron_annotations.tsv", 31718505),
}


def main():
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    for path, (url, size) in FILES.items():
        if path.exists() and path.stat().st_size == size:
            print(f"ok       {path.name}")
            continue
        print(f"download {path.name} ({size / 1e6:.0f} MB)")
        urllib.request.urlretrieve(url, path)
        if path.stat().st_size != size:
            sys.exit(f"size mismatch for {path.name}")


if __name__ == "__main__":
    main()
