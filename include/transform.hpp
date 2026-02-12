#pragma once

#include <torch/torch.h>

#include "orthogonalize.hpp"
#include "wavelet.hpp"

struct WaveletPacketResult {
    torch::Tensor coeffs;     // [batch, max_level, N] or [max_level, N]
    int64_t max_level;
    int64_t signal_length;
};

/// Forward 1-D wavelet packet transform.
/// signal: [N] (unbatched) or [batch, N] (batched).
/// Returns coefficients at every level, stacked into [max_level, N] or [batch, max_level, N].
WaveletPacketResult wavelet_packet_forward_1d(
    torch::Tensor const& signal,
    Wavelet const& wavelet,
    int64_t max_level,
    OrthMethod method = OrthMethod::qr);

/// Inverse 1-D wavelet packet transform from leaf-level coefficients.
/// leaf_coeffs: [N] (unbatched) or [batch, N] (batched).
/// Returns reconstructed signal with same rank as input.
torch::Tensor wavelet_packet_inverse_1d(
    torch::Tensor const& leaf_coeffs,
    Wavelet const& wavelet,
    int64_t max_level,
    OrthMethod method = OrthMethod::qr);

/// Inverse 1-D wavelet packet transform from a forward result (uses leaf level).
torch::Tensor wavelet_packet_inverse_1d(
    WaveletPacketResult const& result,
    Wavelet const& wavelet,
    OrthMethod method = OrthMethod::qr);

/// Permutation from natural to frequency (Gray code) subband order.
/// Returns a 1-D int64 tensor of length 2^level.
torch::Tensor natural_to_freq_permutation(int64_t level);
