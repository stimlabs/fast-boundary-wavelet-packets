#include "orthogonalize.h"

#include <stdexcept>

static torch::TensorOptions sparse_opts(torch::Tensor const& t) {
    return torch::TensorOptions().dtype(t.dtype()).device(t.device());
}

OrthMethod parse_orth_method(std::string const& name) {
    if (name == "qr") return OrthMethod::qr;
    if (name == "gramschmidt") return OrthMethod::gramschmidt;
    throw std::invalid_argument("Unknown orthogonalization method: " + name);
}

torch::Tensor find_boundary_rows(
    torch::Tensor const& matrix,
    int64_t filt_len) {

    auto coalesced = matrix.coalesce();
    auto row_indices = coalesced.indices()[0];
    auto [unique, inverse, counts] = at::unique_consecutive(row_indices,
        /*return_inverse=*/false, /*return_counts=*/true);
    return unique.index({counts != filt_len});
}

torch::Tensor orth_by_qr(
    torch::Tensor const& matrix,
    torch::Tensor const& boundary_rows) {

    if (boundary_rows.numel() == 0) {
        return matrix;
    }

    int64_t const k = boundary_rows.numel();
    int64_t const n = matrix.size(0);
    auto const opts = sparse_opts(matrix);
    auto const long_opts = torch::TensorOptions().dtype(torch::kLong).device(matrix.device());

    // Step 1: Extract boundary rows via sparse selection matrix [k, N].
    auto sel_indices = torch::stack({
        torch::arange(k, long_opts),
        boundary_rows});
    auto selection = torch::sparse_coo_tensor(
        sel_indices, torch::ones(k, opts), {k, n}, opts);

    auto sel = torch::mm(selection, matrix);  // sparse [k, N]

    // Step 2: Dense QR on boundary submatrix.
    auto [q, r] = torch::linalg_qr(sel.to_dense().t());  // Q: [N, k]

    // Step 3: Remove old boundary rows via diagonal removal matrix.
    auto diag_idx = torch::arange(n, long_opts);
    auto diag_vals = torch::ones(n, opts);
    diag_vals.index_fill_(0, boundary_rows, 0.0);
    auto removal = torch::sparse_coo_tensor(
        torch::stack({diag_idx, diag_idx}), diag_vals, {n, n}, opts);

    auto result = torch::mm(removal, matrix);

    // Step 4: Insert new orthogonalized rows from Q.T.
    auto q_t = q.t();  // dense [k, N]
    for (int64_t pos = 0; pos < k; ++pos) {
        auto row = q_t[pos].unsqueeze(0).to_sparse().coalesce();
        auto indices = row.indices().clone();
        indices[0] += boundary_rows[pos].item<int64_t>();
        auto addition = torch::sparse_coo_tensor(
            indices, row.values(), {n, n}, opts);
        result = result + addition;
    }

    return result.coalesce();
}

torch::Tensor orthogonalize(
    torch::Tensor const& matrix,
    int64_t filt_len,
    OrthMethod method) {

    auto boundary = find_boundary_rows(matrix, filt_len);

    if (boundary.numel() == 0) {
        return matrix;
    }

    switch (method) {
        case OrthMethod::qr:
            return orth_by_qr(matrix, boundary);
        case OrthMethod::gramschmidt:
            throw std::runtime_error(
                "Gram-Schmidt orthogonalization not yet implemented");
    }
    throw std::logic_error("Invalid OrthMethod");
}
