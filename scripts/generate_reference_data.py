"""Generate reference data from the Python ptwt library for C++ cross-validation."""

from pathlib import Path
import ptwt
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

WPT_1D_TEST_CASES = [
    # (wavelet, L, max_level)
    ("haar", 8, 3),
    ("haar", 16, 4),
    ("db2", 16, 3),
    ("db3", 16, 2),
    ("db3", 32, 3),
]

WPT_2D_TEST_CASES = [
    # (wavelet, H, W, max_level)
    ("haar", 8, 8, 3),
    ("haar", 16, 16, 4),
    ("haar", 16, 8, 3),
    ("db2", 16, 16, 3),
    ("db3", 16, 16, 2),
]

ORTHOGONALIZATION_METHODS = [("qr", "qr"), ("gramschmidt", "gs")]


def save_tensor(tensor, path):
    # LibTorch pickle_load cannot deserialize sparse tensors, so always save dense.
    if tensor.is_sparse:
        tensor = tensor.to_dense()
    torch.save(tensor.contiguous(), path)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Build raw analysis and synthesis matrices:")
    for wavelet, length in MATRIX_BUILD_CASES:
        a = _construct_a(wavelet, length, dtype=torch.float64)
        s = _construct_s(wavelet, length, dtype=torch.float64)
        save_tensor(a, DATA_DIR / f"{wavelet}_{length}_a.pt")
        save_tensor(s, DATA_DIR / f"{wavelet}_{length}_s.pt")
        print(f"  {wavelet} length={length}: A{list(a.shape)}, S{list(s.shape)}")

    for method_name, method_tag in ORTHOGONALIZATION_METHODS:
        print(f"Build boundary matrices ({method_tag}):")
        for wavelet, length in MATRIX_BUILD_CASES:
            ba = construct_boundary_a(wavelet, length, orthogonalization=method_name, dtype=torch.float64)
            bs = construct_boundary_s(wavelet, length, orthogonalization=method_name, dtype=torch.float64)
            save_tensor(ba, DATA_DIR / f"{wavelet}_{length}_boundary_{method_tag}_a.pt")
            save_tensor(bs, DATA_DIR / f"{wavelet}_{length}_boundary_{method_tag}_s.pt")
            print(f"  {wavelet} length={length}: boundary_A{list(ba.shape)}, boundary_S{list(bs.shape)}")

    torch.manual_seed(42)
    print("Wavelet packet transform reference data:")
    for wavelet, length, max_level in WPT_1D_TEST_CASES:
        signal = torch.randn(length, dtype=torch.float64)
        tag = f"wpt_1D_{wavelet}_{length}_L{max_level}"
        save_tensor(signal, DATA_DIR / f"{tag}_signal.pt")

        for method_name, method_tag in ORTHOGONALIZATION_METHODS:
            wp = ptwt.WaveletPacket(
                data=signal,
                wavelet=wavelet,
                mode="boundary",
                maxlevel=max_level,
                orthogonalization=method_name,
            )
            for level in range(1, max_level + 1):
                nodes = wp.get_level(level, order="freq")
                subbands = [wp[node] for node in nodes]
                tiled = torch.cat(subbands, dim=-1)
                save_tensor(
                    tiled,
                    DATA_DIR / f"{tag}_{method_tag}_l{level}.pt",
                )
                print(f"  {tag} {method_tag} level {level}: {list(tiled.shape)} ({len(nodes)} nodes)")

    print("2D wavelet packet transform reference data:")
    for wavelet, H, W, max_level in WPT_2D_TEST_CASES:
        signal = torch.randn(H, W, dtype=torch.float64)
        tag = f"wpt_2D_{wavelet}_{H}x{W}_L{max_level}"
        save_tensor(signal, DATA_DIR / f"{tag}_signal.pt")

        for method_name, method_tag in ORTHOGONALIZATION_METHODS:
            wp = ptwt.WaveletPacket2D(
                data=signal.unsqueeze(0),  # ptwt expects [batch, H, W]
                wavelet=wavelet,
                mode="boundary",
                maxlevel=max_level,
                orthogonalization=method_name,
                separable=True,  # must match our C++ impl (separable only)
            )
            for level in range(1, max_level + 1):
                freq_grid = wp.get_level(level, order="freq")  # returns list[list[str]] (rows × cols)
                rows = []
                for row_nodes in freq_grid:
                    cols = [wp[node].squeeze(0) for node in row_nodes]
                    rows.append(torch.cat(cols, dim=-1))
                tiled = torch.cat(rows, dim=-2)
                save_tensor(
                    tiled,
                    DATA_DIR / f"{tag}_{method_tag}_l{level}.pt",
                )
                print(f"  {tag} {method_tag} level {level}: {list(tiled.shape)} "
                      f"({len(freq_grid)}x{len(freq_grid[0])} subbands)")


if __name__ == "__main__":
    main()
