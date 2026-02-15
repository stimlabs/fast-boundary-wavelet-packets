#include "sparse_math.hpp"

#include "tensor_util.hpp"

torch::Tensor construct_conv_matrix(
    torch::Tensor const& filter,
    int64_t input_length) {

    int64_t const filter_len = filter.size(0);
    int64_t const filter_offset = filter_len % 2;

    // Sameshift centering: the filter is placed so that the center tap aligns
    // with the diagonal. start_row/stop_row define the valid row range after
    // discarding positions that would extend beyond the signal boundaries.
    int64_t const start_row = filter_len / 2 - 1 + filter_offset;
    int64_t const stop_row = start_row + input_length - 1;

    std::vector<int64_t> rows;
    std::vector<int64_t> cols;
    std::vector<int64_t> filter_indices;

    for (int64_t col = 0; col < input_length; ++col) {
        for (int64_t fi = 0; fi < filter_len; ++fi) {
            int64_t const pos = fi + col;
            if (pos >= start_row && pos <= stop_row) {
                rows.push_back(pos - start_row);
                cols.push_back(col);
                filter_indices.push_back(fi);
            }
        }
    }

    auto const long_opts = long_opts_like(filter);
    auto row_idx = torch::tensor(rows, long_opts);
    auto col_idx = torch::tensor(cols, long_opts);
    auto indices = torch::stack({row_idx, col_idx});

    auto values = filter.index({torch::tensor(filter_indices, long_opts)});

    return torch::sparse_coo_tensor(
        indices, values, {input_length, input_length}, sparse_opts(filter));
}

torch::Tensor construct_strided_conv_matrix(
    torch::Tensor const& filter,
    int64_t input_length,
    int64_t stride) {

    auto conv_matrix = construct_conv_matrix(filter, input_length);

    // Build a selection matrix that picks rows 1, 1+stride, 1+2*stride, ...
    // from the full convolution matrix. This implements the sameshift downsampling:
    // row 1 is the first valid center-aligned output position.
    auto const long_opts = long_opts_like(filter);
    auto select_rows = torch::arange(1, conv_matrix.size(0), stride, long_opts);
    int64_t const n_selected = select_rows.size(0);

    auto sel_indices = torch::stack({
        torch::arange(0, n_selected, long_opts),
        select_rows});
    auto sel_values = torch::ones(n_selected, sparse_opts(filter));

    auto selection_matrix = torch::sparse_coo_tensor(
        sel_indices, sel_values,
        {n_selected, conv_matrix.size(0)},
        sparse_opts(filter));

    return torch::mm(selection_matrix, conv_matrix);
}

torch::Tensor sparse_replace_row(
    torch::Tensor const& matrix,
    int64_t row_index,
    torch::Tensor const& row) {

    int64_t const num_rows = matrix.size(0);
    auto const opts = sparse_opts(matrix);
    auto const long_opts = long_opts_like(matrix);

    // Diagonal removal matrix: identity with 0 at row_index.
    auto diag_idx = torch::arange(num_rows, long_opts);
    auto diag_vals = torch::ones(num_rows, opts);
    diag_vals[row_index] = 0.0;
    auto removal = torch::sparse_coo_tensor(
        torch::stack({diag_idx, diag_idx}), diag_vals, {num_rows, num_rows}, opts);

    auto result = torch::mm(removal, matrix);

    // Addition matrix: place the new row at row_index.
    auto row_c = row.coalesce();
    auto row_indices = row_c.indices();  // [2, nnz] with row dim all 0
    auto shifted_indices = torch::stack({
        row_indices[0] + row_index,
        row_indices[1]});
    auto addition = torch::sparse_coo_tensor(
        shifted_indices, row_c.values(), matrix.sizes(), opts);

    return (result + addition).coalesce();
}

torch::Tensor block_diag_repeat(
    torch::Tensor const& block,
    int64_t count) {

    auto block_c = block.coalesce();
    int64_t const block_size = block_c.size(0);
    auto src_indices = block_c.indices();   // [2, nnz]
    auto src_values = block_c.values();     // [nnz]
    int64_t const nnz = src_values.size(0);

    auto const long_opts = long_opts_like(block);

    // Pre-allocate index/value tensors for all copies at once.
    // Each copy's indices are offset by i * block_size along both row and col axes.
    auto all_rows = torch::empty(nnz * count, long_opts);
    auto all_cols = torch::empty(nnz * count, long_opts);
    auto all_vals = torch::empty(nnz * count, sparse_opts(block));

    auto src_rows = src_indices[0];
    auto src_cols = src_indices[1];

    for (int64_t i = 0; i < count; ++i) {
        int64_t const offset = i * nnz;
        int64_t const block_offset = i * block_size;
        all_rows.slice(0, offset, offset + nnz).copy_(src_rows + block_offset);
        all_cols.slice(0, offset, offset + nnz).copy_(src_cols + block_offset);
        all_vals.slice(0, offset, offset + nnz).copy_(src_values);
    }

    int64_t const total_size = block_size * count;
    return torch::sparse_coo_tensor(
        torch::stack({all_rows, all_cols}), all_vals, {total_size, total_size}, sparse_opts(block));
}
