"""Generate reference data from the Python ptwt library for C++ cross-validation."""

from pathlib import Path
import torch
from ptwt.matmul_transform import _construct_a, _construct_s

DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"

MATRIX_BUILD_CASES = [
    ("haar", 8),
    ("haar", 16),
    ("db2", 8),
    ("db2", 16),
    ("db3", 16),
]


def save_tensor(tensor, path):
    torch.save(tensor, path)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for wavelet, length in MATRIX_BUILD_CASES:
        a = _construct_a(wavelet, length, dtype=torch.float64)
        s = _construct_s(wavelet, length, dtype=torch.float64)
        save_tensor(a, DATA_DIR / f"{wavelet}_{length}_a.pt")
        save_tensor(s, DATA_DIR / f"{wavelet}_{length}_s.pt")
        print(f"  {wavelet} length={length}: A{list(a.shape)}, S{list(s.shape)}")
    print("Done.")


if __name__ == "__main__":
    main()
