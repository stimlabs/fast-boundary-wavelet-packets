#pragma once

#include <torch/torch.h>

#include <string>

/// Supported orthogonalization methods for boundary wavelet filters.
enum class OrthMethod { qr, gramschmidt };

/// Parse an orthogonalization method from a string.
/// Accepted values: "qr", "gramschmidt".
OrthMethod parse_orth_method(std::string const& name);

/// Row indices where non-zero count != filt_len. Returns 1-D int64 tensor.
torch::Tensor find_boundary_rows(
    torch::Tensor const& matrix,
    int64_t filt_len);

/// Replace boundary rows with QR-orthonormalized versions.
/// Clones the matrix; input is not modified.
torch::Tensor orth_by_qr(
    torch::Tensor const& matrix,
    torch::Tensor const& boundary_rows);

/// Top-level dispatch: find boundary rows, orthogonalize by method.
torch::Tensor orthogonalize(
    torch::Tensor const& matrix,
    int64_t filt_len,
    OrthMethod method = OrthMethod::qr);
