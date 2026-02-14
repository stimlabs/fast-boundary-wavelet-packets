"""Benchmark: fbwp (C++ / LibTorch) vs ptwt (Python / PyTorch) wavelet packet transform."""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
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


def time_fn(
    fn, signals: list[torch.Tensor], *args, warmup: int, runs: int, sync_cuda: bool = False,
) -> tuple[float, float, float]:
    """Time a function: returns median wall-clock time in seconds.

    Each iteration uses a fresh signal from *signals* so that no input is read
    from hot cache.  ``signals`` must have length >= warmup + runs.
    """
    idx = 0
    for _ in range(warmup):
        fn(signals[idx], *args)
        if sync_cuda:
            torch.cuda.synchronize()
        idx += 1
    times: list[float] = []
    for _ in range(runs):
        if sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(signals[idx], *args)
        if sync_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        idx += 1
    times: torch.Tensor = torch.tensor(times)
    return torch.median(times).item(), torch.mean(times).item(), torch.std(times).item()


def main(
    *,
    wavelets: list[str],
    signal_lengths: list[int],
    batch_size: int,
    dtype: str,
    device: str | None,
    orth_method: Literal["qr", "gramschmidt"],
    warmup_runs: int,
    timed_runs: int,
    random_seed: int,
    output_file: Path | None,
) -> None:
    torch.manual_seed(random_seed)

    torch_dtype = DTYPE_CHOICES[dtype]
    torch_device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    rtol, atol = TOLERANCES[torch_dtype]
    sync_cuda = torch_device.type == "cuda"
    total_runs = warmup_runs + timed_runs

    print(f"wavelets:                    {' '.join(wavelets)}")
    print(f"signal lengths:              {' '.join(map(str, signal_lengths))}")
    print(f"batch size:                  {batch_size}")
    print(f"dtype:                       {dtype}")
    print(f"device:                      {torch_device}")
    print(f"orthogonalization method:    {orth_method}")
    print(f"warmup / runs:               {warmup_runs} / {timed_runs}")
    print(f"random seed:                 {random_seed}")
    print()

    hdr = (
        f"{'wavelet':<10} {'length':>8} {'level':>5} "
        f"{'ptwt mean':>10} {'± std':>10} "
        f"{'fbwp mean':>10} {'± std':>10} "
        f"{'speedup':>8} {'match':>6}"
    )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    results: list[dict] = []

    for wavelet in wavelets:
        for length in signal_lengths:
            dec_len = pywt.Wavelet(wavelet).dec_len
            max_level = fbwp.compute_max_level(length, dec_len)
            if max_level < 1:
                continue

            signals = [
                torch.randn(batch_size, length, dtype=torch_dtype, device=torch_device)
                for _ in range(total_runs)
            ]

            ptwt_median, ptwt_mean, ptwt_std = time_fn(
                ptwt_forward, signals, wavelet, max_level, orth_method,
                warmup=warmup_runs, runs=timed_runs, sync_cuda=sync_cuda)
            fbwp_median, fbwp_mean, fbwp_std = time_fn(
                fbwp_forward, signals, wavelet, max_level, orth_method,
                warmup=warmup_runs, runs=timed_runs, sync_cuda=sync_cuda)

            check_signal = signals[-1]
            coefficients_ptwt = ptwt_forward(check_signal, wavelet, max_level, orth_method)
            coefficients_fbwp = fbwp_forward(check_signal, wavelet, max_level, orth_method)
            match = torch.allclose(coefficients_ptwt, coefficients_fbwp, rtol=rtol, atol=atol)

            speedup = ptwt_median / fbwp_median if fbwp_median > 0 else float("inf")

            results.append({
                "wavelet": wavelet,
                "length": length,
                "max_level": max_level,
                "ptwt": {"median_s": ptwt_median, "mean_s": ptwt_mean, "std_s": ptwt_std},
                "fbwp": {"median_s": fbwp_median, "mean_s": fbwp_mean, "std_s": fbwp_std},
                "speedup": speedup,
                "match": match,
            })

            print(
                f"{wavelet:<10} {length:>8} {max_level:>5} "
                f"{ptwt_mean * 1e3:>9.2f}ms {ptwt_std * 1e3:>9.2f}ms "
                f"{fbwp_mean * 1e3:>9.2f}ms {fbwp_std * 1e3:>9.2f}ms "
                f"{speedup:>7.2f}x {'OK' if match else 'FAIL':>5}"
            )

    print(sep)

    benchmark = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "config": {
            "wavelets": wavelets,
            "signal_lengths": signal_lengths,
            "batch_size": batch_size,
            "dtype": dtype,
            "device": str(torch_device),
            "orth_method": orth_method,
            "warmup_runs": warmup_runs,
            "timed_runs": timed_runs,
            "random_seed": random_seed,
        },
        "results": results,
    }

    if output_file is not None:
        with output_file.open("w") as f:
            json.dump(benchmark, f, indent=2)
        print(f"results saved to {output_file!s}")


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
        default=[1024, 2048, 4096, 8192],
        help="signal lengths to benchmark (default: 1024 .. 8192)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="number of signals per batch, i.e. first dimension of the input tensor (default: 1)",
    )
    parser.add_argument(
        "--dtype", choices=list(DTYPE_CHOICES), default="float64",
        help="tensor dtype (default: float64)",
    )
    parser.add_argument(
        "--device", default=None,
        help="torch device (default: cuda if available, else cpu)",
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
    parser.add_argument(
        "--output-file", "-O", type=Path, default=None,
        help="path to save benchmark results as JSON (default: no save)",
    )

    main(**vars(parser.parse_args()))
