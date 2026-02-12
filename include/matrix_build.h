#pragma once

#include <torch/torch.h>

#include "wavelet.h"

/// Build the raw analysis matrix A for a single decomposition level.
/// Top length/2 rows: lowpass (dec_lo), bottom length/2 rows: highpass (dec_hi).
/// Shape: (length, length).
torch::Tensor construct_a(
    Wavelet const& wavelet,
    int64_t length,
    torch::TensorOptions const& opts);

/// Build the raw synthesis matrix S for a single decomposition level.
/// S = cat(strided_conv(flip(rec_lo)), strided_conv(flip(rec_hi))).T
/// Shape: (length, length).
torch::Tensor construct_s(
    Wavelet const& wavelet,
    int64_t length,
    torch::TensorOptions const& opts);
