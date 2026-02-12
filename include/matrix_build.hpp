#pragma once

#include <torch/torch.h>

#include "orthogonalize.hpp"
#include "wavelet.hpp"

/// Build the raw analysis matrix A for a single decomposition level.
/// Top length/2 rows: lowpass (dec_lo), bottom length/2 rows: highpass (dec_hi).
/// Returns a sparse COO tensor of shape (length, length).
torch::Tensor construct_a(
    Wavelet const& wavelet,
    int64_t length,
    torch::TensorOptions const& opts);

/// Build the raw synthesis matrix S for a single decomposition level.
/// S = cat(strided_conv(flip(rec_lo)), strided_conv(flip(rec_hi))).T
/// Returns a sparse COO tensor of shape (length, length).
torch::Tensor construct_s(
    Wavelet const& wavelet,
    int64_t length,
    torch::TensorOptions const& opts);

/// Build the boundary-orthogonalized analysis matrix.
/// Returns a sparse COO tensor of shape (length, length).
torch::Tensor construct_boundary_a(
    Wavelet const& wavelet,
    int64_t length,
    torch::TensorOptions const& opts,
    OrthMethod method = OrthMethod::qr);

/// Build the boundary-orthogonalized synthesis matrix.
/// Returns a sparse COO tensor of shape (length, length).
torch::Tensor construct_boundary_s(
    Wavelet const& wavelet,
    int64_t length,
    torch::TensorOptions const& opts,
    OrthMethod method = OrthMethod::qr);
