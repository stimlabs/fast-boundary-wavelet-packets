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

/// Replace a single row in a sparse matrix. Returns a new sparse matrix.
/// The row tensor must be sparse with shape [1, cols].
torch::Tensor sparse_replace_row(
    torch::Tensor const& matrix,
    int64_t row_index,
    torch::Tensor const& row);

/// Repeat a sparse square block along the diagonal.
/// Given a sparse [M, M] block and count k, returns a sparse [M*k, M*k] tensor.
torch::Tensor block_diag_repeat(
    torch::Tensor const& block,
    int64_t count);
