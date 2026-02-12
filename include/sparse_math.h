#pragma once

#include <torch/torch.h>

/// Build a sparse convolution matrix (sameshift mode).
/// Result is a square (input_length x input_length) sparse COO tensor.
torch::Tensor construct_conv_matrix(
    torch::Tensor const& filter,
    int64_t input_length);

/// Build a strided sparse convolution matrix (sameshift mode).
/// Selects every stride-th row starting at row 1 from the full conv matrix.
/// Result shape: (input_length / stride, input_length).
torch::Tensor construct_strided_conv_matrix(
    torch::Tensor const& filter,
    int64_t input_length,
    int64_t stride);
