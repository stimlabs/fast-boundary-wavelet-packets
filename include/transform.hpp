#pragma once

#include <string>

#include <torch/torch.h>

#include "orthogonalize.hpp"

/// Compute the maximum feasible wavelet packet decomposition level.
/// Largest L where: N % 2^L == 0 AND N / 2^(L-1) >= dec_len.
int64_t compute_max_level(int64_t signal_length, int64_t dec_len);

/// Forward 1-D wavelet packet transform.
/// Returns tensor with a new max_level dimension inserted at position `dim`.
/// The analyzed dimension shifts to `dim + 1` (for inputs with ndim > 1).
/// Example: input_signal [batch, N] with dim=-1 → [batch, max_level, N].
torch::Tensor wavelet_packet_forward_1d(
    torch::Tensor const& input_signal,
    std::string const& wavelet_name,
    int64_t dim = -1,
    int64_t max_level = -1,
    OrthMethod orth_method = OrthMethod::qr);

/// Inverse 1-D wavelet packet transform from leaf-level coefficients.
/// leaf_coeffs has the same shape as the analyzed dimension of the input.
torch::Tensor wavelet_packet_inverse_1d(
    torch::Tensor const& leaf_coeffs,
    std::string const& wavelet_name,
    int64_t dim = -1,
    int64_t max_level = -1,
    OrthMethod orth_method = OrthMethod::qr);

/// Permutation from natural to frequency (Gray code) subband order.
/// Returns a 1-D int64 tensor of length 2^level.
torch::Tensor natural_to_freq_permutation(int64_t level);
