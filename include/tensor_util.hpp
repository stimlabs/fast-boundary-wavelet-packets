#pragma once

#include <torch/torch.h>

/// Build TensorOptions with only dtype and device from an existing tensor.
/// Necessary because tensor.options() includes layout (e.g. Strided), which
/// is incompatible with torch::sparse_coo_tensor.
inline torch::TensorOptions sparse_opts(torch::Tensor const& t) {
    return torch::TensorOptions().dtype(t.dtype()).device(t.device());
}

/// Build int64 TensorOptions on the same device as the given tensor.
/// Used for index tensors (row/col indices) in sparse matrix construction.
inline torch::TensorOptions long_opts_like(torch::Tensor const& t) {
    return torch::TensorOptions().dtype(torch::kLong).device(t.device());
}
