#include "sparse_math.h"

static torch::TensorOptions sparse_opts(torch::Tensor const& t) {
    return torch::TensorOptions().dtype(t.dtype()).device(t.device());
}

torch::Tensor construct_conv_matrix(
    torch::Tensor const& filter,
    int64_t input_length) {

    int64_t const filter_len = filter.size(0);
    int64_t const filter_offset = filter_len % 2;
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

    auto const long_opts = torch::TensorOptions().dtype(torch::kLong);
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

    // sameshift: select rows starting at 1 with given stride
    auto select_rows = torch::arange(1, conv_matrix.size(0), stride);
    int64_t const n_selected = select_rows.size(0);

    auto sel_indices = torch::stack({
        torch::arange(0, n_selected),
        select_rows});
    auto sel_values = torch::ones(n_selected, sparse_opts(filter));

    auto selection_matrix = torch::sparse_coo_tensor(
        sel_indices, sel_values,
        {n_selected, conv_matrix.size(0)},
        sparse_opts(filter));

    return torch::mm(selection_matrix, conv_matrix);
}
