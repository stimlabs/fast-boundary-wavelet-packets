#include "orthogonalize.hpp"

#include "sparse_math.hpp"
#include "tensor_util.hpp"

#include <stdexcept>

OrthMethod parse_orth_method(std::string const& name) {
    if (name == "qr") return OrthMethod::qr;
    if (name == "gramschmidt") return OrthMethod::gramschmidt;
    throw std::invalid_argument("Unknown orthogonalization method: " + name);
}

/// Boundary rows are those whose number of non-zeros differs from the standard
/// interior filter length. Interior rows each have exactly filt_len non-zeros;
/// boundary rows near the signal edges have fewer.
torch::Tensor find_boundary_rows(
    torch::Tensor const& matrix,
    int64_t filt_len) {

    auto coalesced = matrix.coalesce();
    auto row_indices = coalesced.indices()[0];
    auto [unique, inverse, counts] = at::unique_consecutive(row_indices,
        /*return_inverse=*/false, /*return_counts=*/true);
    return unique.index({counts != filt_len});
}

/// QR orthogonalization strategy:
///   1. Extract the small boundary submatrix (num_boundary x N).
///   2. Compute dense QR on its transpose to get an orthonormal basis for the
///      column space — this is efficient because num_boundary << N.
///   3. Remove the old boundary rows from the matrix (zero them out).
///   4. Insert the new orthonormalized rows from Q^T.
torch::Tensor orth_by_qr(
    torch::Tensor const& matrix,
    torch::Tensor const& boundary_rows) {

    if (boundary_rows.numel() == 0) {
        return matrix;
    }

    int64_t const num_boundary = boundary_rows.numel();
    int64_t const num_rows = matrix.size(0);
    auto const opts = sparse_opts(matrix);
    auto const long_opts = long_opts_like(matrix);

    // Step 1: Extract boundary rows via sparse selection matrix [num_boundary, num_rows].
    auto sel_indices = torch::stack({
        torch::arange(num_boundary, long_opts),
        boundary_rows});
    auto selection = torch::sparse_coo_tensor(
        sel_indices, torch::ones(num_boundary, opts), {num_boundary, num_rows}, opts);

    auto boundary_submatrix = torch::mm(selection, matrix);  // sparse [num_boundary, N]

    // Step 2: Dense QR on boundary submatrix — small (num_boundary rows), so dense is fine.
    auto [q, r] = torch::linalg_qr(boundary_submatrix.to_dense().t());  // Q: [N, num_boundary]

    // Step 3: Remove old boundary rows via diagonal removal matrix.
    auto diag_idx = torch::arange(num_rows, long_opts);
    auto diag_vals = torch::ones(num_rows, opts);
    diag_vals.index_fill_(0, boundary_rows, 0.0);
    auto removal = torch::sparse_coo_tensor(
        torch::stack({diag_idx, diag_idx}), diag_vals, {num_rows, num_rows}, opts);

    auto result = torch::mm(removal, matrix);

    // Step 4: Insert new orthogonalized rows from Q^T.
    auto q_t = q.t();  // dense [num_boundary, N]
    for (int64_t pos = 0; pos < num_boundary; ++pos) {
        auto row = q_t[pos].unsqueeze(0).to_sparse().coalesce();
        auto indices = row.indices().clone();
        indices[0] += boundary_rows[pos].item<int64_t>();
        auto addition = torch::sparse_coo_tensor(
            indices, row.values(), {num_rows, num_rows}, opts);
        result = result + addition;
    }

    return result.coalesce();
}

/// Classical Gram-Schmidt orthogonalization, fully sparse:
/// For each boundary row, subtract the projections onto all previously
/// orthogonalized boundary rows, then normalize to unit length.
torch::Tensor orth_by_gram_schmidt(
    torch::Tensor const& matrix,
    torch::Tensor const& boundary_rows) {

    if (boundary_rows.numel() == 0) {
        return matrix;
    }

    int64_t const num_rows = matrix.size(0);
    auto const opts = sparse_opts(matrix);
    auto const long_opts = long_opts_like(matrix);

    auto result = matrix;
    std::vector<int64_t> processed_rows;

    for (int64_t i = 0; i < boundary_rows.numel(); ++i) {
        int64_t const row_idx = boundary_rows[i].item<int64_t>();

        // Extract current row via sparse selection matrix -> [1, N]
        auto sel_indices = torch::stack({
            torch::zeros(1, long_opts),
            torch::tensor({row_idx}, long_opts)});
        auto selection = torch::sparse_coo_tensor(
            sel_indices, torch::ones(1, opts), {1, num_rows}, opts);
        auto current_row = torch::mm(selection, result);  // sparse [1, N]

        // Accumulate projections onto all previously processed boundary rows.
        auto projection_sum = torch::sparse_coo_tensor(
            torch::empty({2, 0}, long_opts),
            torch::empty(0, opts),
            {1, matrix.size(1)}, opts);

        for (int64_t processed_row_idx : processed_rows) {
            // Extract the already-orthogonalized row
            auto done_sel_indices = torch::stack({
                torch::zeros(1, long_opts),
                torch::tensor({processed_row_idx}, long_opts)});
            auto done_selection = torch::sparse_coo_tensor(
                done_sel_indices, torch::ones(1, opts), {1, num_rows}, opts);
            auto done_row = torch::mm(done_selection, result);  // sparse [1, N]

            // Inner product: <current_row, done_row> = current_row @ done_row^T
            auto inner_product = torch::mm(current_row, done_row.t()).coalesce();
            double scalar = 0.0;
            if (inner_product.values().numel() > 0) {
                scalar = inner_product.values()[0].item<double>();
            }
            projection_sum = (projection_sum + scalar * done_row).coalesce();
        }

        // Orthogonalize (subtract projections) and normalize to unit length.
        auto orthogonal = (current_row.to_dense() - projection_sum.to_dense()).to_sparse().coalesce();
        auto length = torch::native_norm(orthogonal);
        auto orthonormal = (orthogonal / length).coalesce();

        result = sparse_replace_row(result, row_idx, orthonormal);
        processed_rows.push_back(row_idx);
    }

    return result;
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
            return orth_by_gram_schmidt(matrix, boundary);
    }
    throw std::logic_error("Invalid OrthMethod");
}
