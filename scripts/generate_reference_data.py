"""Generate reference data from the Python ptwt library for C++ cross-validation."""

from pathlib import Path
import torch
from ptwt.matmul_transform import (
    _construct_a,
    _construct_s,
    construct_boundary_a,
    construct_boundary_s,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"

MATRIX_BUILD_CASES = [
    ("haar", 8),
    ("haar", 16),
    ("db2", 8),
    ("db2", 16),
    ("db3", 16),
]


def save_tensor(tensor, path):
    # LibTorch pickle_load cannot deserialize sparse tensors, so always save dense.
    torch.save(tensor.to_dense(), path)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Build raw analysis and synthesis matrices:")
    for wavelet, length in MATRIX_BUILD_CASES:
        a = _construct_a(wavelet, length, dtype=torch.float64)
        s = _construct_s(wavelet, length, dtype=torch.float64)
        save_tensor(a, DATA_DIR / f"{wavelet}_{length}_a.pt")
        save_tensor(s, DATA_DIR / f"{wavelet}_{length}_s.pt")
        print(f"  {wavelet} length={length}: A{list(a.shape)}, S{list(s.shape)}")
    print("Build boundary matrices (QR):")
    for wavelet, length in MATRIX_BUILD_CASES:
        ba = construct_boundary_a(wavelet, length, orthogonalization="qr", dtype=torch.float64)
        bs = construct_boundary_s(wavelet, length, orthogonalization="qr", dtype=torch.float64)
        save_tensor(ba, DATA_DIR / f"{wavelet}_{length}_boundary_a.pt")
        save_tensor(bs, DATA_DIR / f"{wavelet}_{length}_boundary_s.pt")
        print(f"  {wavelet} length={length}: boundary_A{list(ba.shape)}, boundary_S{list(bs.shape)}")
    print("Done.")


if __name__ == "__main__":
    main()
