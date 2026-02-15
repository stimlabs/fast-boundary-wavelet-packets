"""Type stubs for the fbwp C++ extension module."""

import torch

def wavelet_packet_forward_1d(
    input_signal: torch.Tensor,
    wavelet_name: str,
    dim: int = -1,
    max_level: int = -1,
    orth_method: str = "qr",
) -> torch.Tensor:
    """Forward 1-D wavelet packet transform.

    Returns a tensor with a new ``max_level`` dimension inserted at
    position *dim*.  The analyzed dimension shifts to ``dim + 1``
    (for inputs with ndim > 1).

    Example: ``input_signal [batch, N]`` with ``dim=-1`` gives
    ``[batch, max_level, N]``.
    """
    ...

def wavelet_packet_inverse_1d(
    leaf_coeffs: torch.Tensor,
    wavelet_name: str,
    dim: int = -1,
    max_level: int = -1,
    orth_method: str = "qr",
) -> torch.Tensor:
    """Inverse 1-D wavelet packet transform from leaf-level coefficients."""
    ...

def wavelet_packet_forward_2d(
    input_signal: torch.Tensor,
    wavelet_name: str,
    dims: tuple[int, int] = (-2, -1),
    max_level: int = -1,
    orth_method: str = "qr",
) -> torch.Tensor:
    """Forward 2-D wavelet packet transform (separable)."""
    ...

def wavelet_packet_inverse_2d(
    leaf_coeffs: torch.Tensor,
    wavelet_name: str,
    dims: tuple[int, int] = (-2, -1),
    max_level: int = -1,
    orth_method: str = "qr",
) -> torch.Tensor:
    """Inverse 2-D wavelet packet transform from leaf-level coefficients (separable)."""
    ...

def compute_max_level(signal_length: int, dec_len: int) -> int:
    """Compute the maximum feasible wavelet packet decomposition level.

    Largest *L* where ``signal_length % 2^L == 0`` and
    ``signal_length / 2^(L-1) >= dec_len``.
    """
    ...
