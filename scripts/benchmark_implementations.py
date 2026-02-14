"""Benchmark: fbwp (C++ / LibTorch) vs ptwt (Python / PyTorch) wavelet packet transform."""

import argparse
import time
from typing import Literal

import fbwp
import ptwt
import pywt
import torch

# tolerances from https://docs.pytorch.org/docs/stable/testing.html
TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float32: (1.3e-6, 1e-5),
    torch.float64: (1e-7, 1e-7),
}

DTYPE_CHOICES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def ptwt_forward(
        signal: torch.Tensor, wavelet: str, max_level: int, orth_method: Literal["qr", "gramschmidt"]
) -> torch.Tensor:
    """Run ptwt forward wavelet packet transform, return leaf-level coefficients."""
    wp = ptwt.WaveletPacket(
        data=signal,
        wavelet=wavelet,
        mode="boundary",
        maxlevel=max_level,
        orthogonalization=orth_method,
    )
    nodes = wp.get_level(max_level, order="freq")
    return torch.cat([wp[n] for n in nodes], dim=-1)


def fbwp_forward(
        signal: torch.Tensor, wavelet: str, max_level: int, orth_method: Literal["qr", "gramschmidt"]
) -> torch.Tensor:
    """Run fbwp forward wavelet packet transform, return leaf-level coefficients."""
    coeffs = fbwp.wavelet_packet_forward_1d(
        signal, wavelet, -1, max_level, orth_method
    )
    # coeffs shape: [..., max_level, N] — extract the last level
    return coeffs[..., -1, :]


def time_fn(fn, *args, warmup: int, runs: int) -> float:
    """Time a function: returns median wall-clock time in seconds."""
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times.sort()
    return times[len(times) // 2]


def main(
    *,
    wavelets: list[str],
    signal_lengths: list[int],
    dtype: str,
    orth_method: Literal["qr", "gramschmidt"],
    warmup_runs: int,
    timed_runs: int,
    random_seed: int,
) -> None:
    torch_dtype = DTYPE_CHOICES[dtype]
    rtol, atol = TOLERANCES[torch_dtype]

    hdr = (
        f"{'wavelet':<10} {'length':>8} {'level':>5} "
        f"{'ptwt (ms)':>10} {'fbwp (ms)':>10} {'speedup':>8} {'match':>6}"
    )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    torch.manual_seed(random_seed)

    for wavelet in wavelets:
        for length in signal_lengths:
            signal = torch.randn(1, length, dtype=torch_dtype)

            dec_len = pywt.Wavelet(wavelet).dec_len
            max_level = fbwp.compute_max_level(length, dec_len)
            if max_level < 1:
                continue

            t_ptwt = time_fn(ptwt_forward, signal, wavelet, max_level, orth_method,
                             warmup=warmup_runs, runs=timed_runs)
            t_fbwp = time_fn(fbwp_forward, signal, wavelet, max_level, orth_method,
                             warmup=warmup_runs, runs=timed_runs)

            c_ptwt = ptwt_forward(signal, wavelet, max_level, orth_method)
            c_fbwp = fbwp_forward(signal, wavelet, max_level, orth_method)
            match = torch.allclose(c_ptwt, c_fbwp, rtol=rtol, atol=atol)

            speedup = t_ptwt / t_fbwp if t_fbwp > 0 else float("inf")
            print(
                f"{wavelet:<10} {length:>8} {max_level:>5} "
                f"{t_ptwt * 1e3:>10.2f} {t_fbwp * 1e3:>10.2f} "
                f"{speedup:>7.2f}x {'OK' if match else 'FAIL':>5}"
            )

    print(sep)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark fbwp (C++) vs ptwt (Python) wavelet packet transforms",
    )
    parser.add_argument(
        "--wavelets", nargs="+", default=["haar", "db2", "db3"],
        help="wavelet names to benchmark (default: haar db2 db3)",
    )
    parser.add_argument(
        "--signal-lengths", nargs="+", type=int,
        default=[1024, 2048, 4096, 8192, 16384, 32768, 65536],
        help="signal lengths to benchmark (default: 1024 .. 65536)",
    )
    parser.add_argument(
        "--dtype", choices=list(DTYPE_CHOICES), default="float64",
        help="tensor dtype (default: float64)",
    )
    parser.add_argument(
        "--orth-method", choices=["qr", "gramschmidt"], default="qr",
        help="orthogonalization method (default: qr)",
    )
    parser.add_argument(
        "--warmup-runs", type=int, default=5,
        help="number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--timed-runs", type=int, default=20,
        help="number of timed iterations (default: 20)",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="random seed (default: 42)",
    )

    main(**vars(parser.parse_args()))
