#include "transform.hpp"

#include "matrix_build.hpp"
#include "sparse_math.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

// ========================== Helper functions ==========================

/// Size of each subband before the split at this level: signal_length / 2^(level-1).
static int64_t subband_size_before_split(int64_t signal_length, int64_t level) {
    return signal_length / (1LL << (level - 1));
}

/// Number of subbands that exist before the split at this level: 2^(level-1).
static int64_t num_subbands_before_split(int64_t level) {
    return 1LL << (level - 1);
}

/// Number of subbands that exist after the split at this level: 2^level.
static int64_t num_subbands_after_split(int64_t level) {
    return 1LL << level;
}

/// Invert a permutation: given perm[i] = j, compute inv[j] = i.
/// Uses scatter_ to write index i at position perm[i].
static torch::Tensor invert_permutation(torch::Tensor const& perm) {
    auto inv = torch::empty_like(perm);
    inv.scatter_(0, perm, torch::arange(perm.size(0), torch::kLong));
    return inv;
}

/// Permute subbands in a [N, batch] column-major tensor.
/// perm is [num_subbands] giving the source subband index for each destination position.
static torch::Tensor permute_subbands(
    torch::Tensor const& data,
    int64_t num_subbands,
    torch::Tensor const& perm) {

    int64_t const total_length = data.size(0);
    int64_t const subband_size = total_length / num_subbands;
    auto result = torch::empty_like(data);
    for (int64_t i = 0; i < num_subbands; ++i) {
        int64_t const src = perm[i].item<int64_t>();
        result.slice(0, i * subband_size, (i + 1) * subband_size)
            .copy_(data.slice(0, src * subband_size, (src + 1) * subband_size));
    }
    return result;
}

int64_t compute_max_level(int64_t signal_length, int64_t dec_len) {
    int64_t L = 0;
    int64_t divisor = 1;
    while (signal_length % (divisor * 2) == 0 && signal_length / divisor >= dec_len) {
        ++L;
        divisor *= 2;
    }
    return L;
}

/// Validate wavelet packet arguments for one or more signal dimensions.
/// For 1-D: pass a single-element vector {N}.
/// For 2-D: pass a two-element vector {H, W}.
/// On success, max_level is set (auto-computed if it was -1).
static void validate_packet_args(
    std::vector<int64_t> const& signal_dims,
    int64_t& max_level,
    int64_t dec_len) {

    if (max_level == -1) {
        // Auto-compute: take the minimum feasible level across all dimensions.
        max_level = std::numeric_limits<int64_t>::max();
        for (int64_t dim_size : signal_dims) {
            max_level = std::min(max_level, compute_max_level(dim_size, dec_len));
        }
        if (max_level < 1) {
            std::string shape_str;
            for (size_t i = 0; i < signal_dims.size(); ++i) {
                if (i > 0) shape_str += ", ";
                shape_str += std::to_string(signal_dims[i]);
            }
            throw std::invalid_argument(
                "Cannot auto-compute max_level: signal shape (" + shape_str +
                ") with filter length " + std::to_string(dec_len) + " yields level 0");
        }
        return;
    }

    if (max_level < 1) {
        throw std::invalid_argument("max_level must be >= 1, got " + std::to_string(max_level));
    }

    int64_t const divisor = 1LL << max_level;
    for (size_t i = 0; i < signal_dims.size(); ++i) {
        int64_t const dim_size = signal_dims[i];
        if (dim_size % divisor != 0) {
            std::string label = signal_dims.size() == 1 ? "Signal length" :
                                (i == 0 ? "Height" : "Width");
            throw std::invalid_argument(
                label + " " + std::to_string(dim_size) +
                " is not divisible by 2^max_level = " + std::to_string(divisor));
        }
        // Smallest subband at the deepest split must be >= dec_len
        int64_t const min_subband = subband_size_before_split(dim_size, max_level);
        if (min_subband < dec_len) {
            std::string label = signal_dims.size() == 1 ? "Subband size" :
                                (i == 0 ? "Row subband size" : "Column subband size");
            throw std::invalid_argument(
                label + " " + std::to_string(min_subband) +
                " at level " + std::to_string(max_level) +
                " is smaller than filter length " + std::to_string(dec_len));
        }
    }
}

/// Resolve a possibly-negative dim index and validate it.
static int64_t resolve_dim(int64_t dim, int64_t ndim) {
    int64_t const resolved = dim < 0 ? dim + ndim : dim;
    if (resolved < 0 || resolved >= ndim) {
        throw std::invalid_argument(
            "dim " + std::to_string(dim) + " out of range for tensor with " +
            std::to_string(ndim) + " dimensions");
    }
    return resolved;
}

// ========================== 1-D transforms ==========================

torch::Tensor wavelet_packet_forward_1d(
    torch::Tensor const& input_signal,
    Wavelet const& wavelet,
    int64_t dim,
    int64_t max_level,
    OrthMethod orth_method) {

    // Resolve dim and move analyzed dimension to last position.
    int64_t const ndim = input_signal.dim();
    if (ndim < 1) {
        throw std::invalid_argument("input_signal must have at least 1 dimension");
    }
    int64_t const resolved_dim = resolve_dim(dim, ndim);

    auto x = torch::movedim(input_signal, resolved_dim, -1);  // [..., N]
    auto const orig_shape = x.sizes().vec();                    // shape with N at the end
    int64_t const N = orig_shape.back();

    validate_packet_args({N}, max_level, wavelet.dec_len());

    // Flatten leading dims into a single batch dim -> [batch, N]
    auto flat = x.reshape({-1, N});

    auto const opts = torch::TensorOptions().dtype(flat.dtype()).device(flat.device());

    // Working state: [N, batch] for efficient sparse mm (sparse acts on leading dim)
    auto subbands = flat.t().contiguous();

    // Collect each level's output
    std::vector<torch::Tensor> level_outputs;
    level_outputs.reserve(max_level);

    for (int64_t level = 1; level <= max_level; ++level) {
        int64_t const M = subband_size_before_split(N, level);
        int64_t const k = num_subbands_before_split(level);

        // Build boundary-orthogonalized analysis matrix, replicated across all subbands.
        auto A_block = construct_boundary_a(wavelet, M, opts, orth_method);
        if (k > 1) {
            A_block = block_diag_repeat(A_block, k);
        }

        // Apply analysis: splits each subband into lowpass + highpass halves.
        subbands = torch::mm(A_block, subbands);

        // Permute from natural to frequency (Gray code) order for output.
        int64_t const num_subbands = num_subbands_after_split(level);
        auto perm = natural_to_freq_permutation(level);
        auto freq_ordered = permute_subbands(subbands, num_subbands, perm);
        level_outputs.push_back(freq_ordered.t().contiguous());  // [batch, N]
    }

    // Stack levels -> [batch, max_level, N]
    auto coeffs = torch::stack(level_outputs, /*dim=*/1);

    // Unflatten batch dims back to original shape: [..., N] -> [..., max_level, N]
    std::vector<int64_t> result_shape(orig_shape.begin(), orig_shape.end() - 1);
    result_shape.push_back(max_level);
    result_shape.push_back(N);
    auto result = coeffs.reshape(result_shape);  // [..., max_level, N]

    // Restore dimension ordering: move max_level and N from trailing positions
    // back to resolved_dim and resolved_dim+1.
    result = torch::movedim(result, std::vector<int64_t>{ndim - 1, ndim},
                                    std::vector<int64_t>{resolved_dim, resolved_dim + 1});

    return result;
}

torch::Tensor wavelet_packet_inverse_1d(
    torch::Tensor const& leaf_coeffs,
    Wavelet const& wavelet,
    int64_t dim,
    int64_t max_level,
    OrthMethod orth_method) {

    // Resolve dim and move analyzed dimension to last position.
    int64_t const ndim = leaf_coeffs.dim();
    if (ndim < 1) {
        throw std::invalid_argument("leaf_coeffs must have at least 1 dimension");
    }
    int64_t const resolved_dim = resolve_dim(dim, ndim);

    auto x = torch::movedim(leaf_coeffs, resolved_dim, -1);  // [..., N]
    auto const orig_shape = x.sizes().vec();
    int64_t const N = orig_shape.back();

    validate_packet_args({N}, max_level, wavelet.dec_len());

    // Flatten leading dims into a single batch dim -> [batch, N]
    auto flat = x.reshape({-1, N});

    auto const opts = torch::TensorOptions().dtype(flat.dtype()).device(flat.device());

    // Working state: [N, batch]
    auto subbands = flat.t().contiguous();

    // Input is in frequency order; convert to natural for the first iteration.
    {
        int64_t const num_subbands = num_subbands_after_split(max_level);
        auto perm = natural_to_freq_permutation(max_level);
        auto inv_perm = invert_permutation(perm);
        subbands = permute_subbands(subbands, num_subbands, inv_perm);
    }

    for (int64_t level = max_level; level >= 1; --level) {
        int64_t const M = subband_size_before_split(N, level);
        int64_t const k = num_subbands_before_split(level);

        // Build boundary-orthogonalized synthesis matrix, replicated across all subbands.
        auto S_block = construct_boundary_s(wavelet, M, opts, orth_method);
        if (k > 1) {
            S_block = block_diag_repeat(S_block, k);
        }

        // Apply synthesis: recombines lowpass + highpass halves back into parent subbands.
        subbands = torch::mm(S_block, subbands);
    }

    auto result = subbands.t().contiguous();  // [batch, N]

    // Unflatten batch dims back to original shape
    std::vector<int64_t> result_shape(orig_shape.begin(), orig_shape.end() - 1);
    result_shape.push_back(N);
    result = result.reshape(result_shape);  // [..., N]

    // Restore dimension ordering: move N from trailing position back to resolved_dim.
    result = torch::movedim(result, -1, resolved_dim);

    return result;
}

// ========================== 2-D helpers ==========================
//
// The 2-D wavelet packet uses a separable approach: two independent 1-D
// sparse matrix multiplications per level (one along rows, one along cols).
// This reuses the same boundary-filter matrices as the 1-D transform.
//
// A subtlety compared to 1-D: the reference library (ptwt, following pywt)
// uses axis 0 = horizontal, axis 1 = vertical. This means the frequency-
// ordered tiling transposes the subband grid relative to the raw separable
// output. See permute_subbands_2d and CLAUDE.md for details.

/// Apply a sparse [H, H] matrix along the row (height) dimension of [batch, H, W].
/// Each column's H-dimensional vector is independently transformed by the matrix.
/// The permute+reshape moves H to the leading dim so sparse mm can act on it,
/// since sparse mm only operates on the leading (row) dimension.
static torch::Tensor apply_matrix_along_rows(
    torch::Tensor const& matrix,
    torch::Tensor const& data) {

    int64_t const batch = data.size(0);
    int64_t const H = data.size(1);
    int64_t const W = data.size(2);
    // [batch, H, W] -> [H, batch, W] -> [H, batch*W] -> mm -> reshape back
    auto col_major = data.permute({1, 0, 2}).reshape({H, batch * W});
    auto result = torch::mm(matrix, col_major);
    return result.reshape({H, batch, W}).permute({1, 0, 2}).contiguous();
}

/// Apply a sparse [W, W] matrix along the column (width) dimension of [batch, H, W].
/// Each row's W-dimensional vector is independently transformed by the matrix.
/// Same permute+reshape strategy as apply_matrix_along_rows but for the W axis.
static torch::Tensor apply_matrix_along_cols(
    torch::Tensor const& matrix,
    torch::Tensor const& data) {

    int64_t const batch = data.size(0);
    int64_t const H = data.size(1);
    int64_t const W = data.size(2);
    // [batch, H, W] -> [W, batch, H] -> [W, batch*H] -> mm -> reshape back
    auto col_major = data.permute({2, 0, 1}).reshape({W, batch * H});
    auto result = torch::mm(matrix, col_major);
    return result.reshape({W, batch, H}).permute({1, 2, 0}).contiguous();
}

/// Permute 2D subbands in a [batch, H, W] tensor from natural to frequency order
/// (or vice versa, when called with an inverse permutation).
///
/// Unlike the 1-D case, the permutation is NOT an independent reordering of
/// row-blocks and column-blocks. The pywt convention (axis 0 = horizontal,
/// axis 1 = vertical) means vertical frequency determines the tiled row
/// position and horizontal frequency determines the tiled column position.
/// Since the raw separable transform tiles as grid[horiz][vert], the mapping
/// to the frequency grid[vert][horiz] requires a transposition:
///
///   freq(fr, fc) = raw(perm[fc], perm[fr])    <- note swapped indices
///
/// At level 1 with identity perm, this swaps the off-diagonal quadrants:
///   raw [LL|LH; HL|HH] -> freq [LL|HL; LH|HH]
static torch::Tensor permute_subbands_2d(
    torch::Tensor const& data,
    int64_t num_subbands,
    torch::Tensor const& perm) {

    int64_t const H = data.size(1);
    int64_t const W = data.size(2);
    int64_t const block_height = H / num_subbands;
    int64_t const block_width = W / num_subbands;

    auto result = torch::empty_like(data);
    for (int64_t freq_row = 0; freq_row < num_subbands; ++freq_row) {
        int64_t const src_col = perm[freq_row].item<int64_t>();  // vert freq -> raw col block
        for (int64_t freq_col = 0; freq_col < num_subbands; ++freq_col) {
            int64_t const src_row = perm[freq_col].item<int64_t>();  // horiz freq -> raw row block
            result.slice(1, freq_row * block_height, (freq_row + 1) * block_height)
                  .slice(2, freq_col * block_width, (freq_col + 1) * block_width)
                  .copy_(data.slice(1, src_row * block_height, (src_row + 1) * block_height)
                             .slice(2, src_col * block_width, (src_col + 1) * block_width));
        }
    }
    return result;
}

// ========================== 2-D transforms ==========================

torch::Tensor wavelet_packet_forward_2d(
    torch::Tensor const& input_signal,
    Wavelet const& wavelet,
    std::array<int64_t, 2> dims,
    int64_t max_level,
    OrthMethod orth_method) {

    int64_t const ndim = input_signal.dim();
    if (ndim < 2) {
        throw std::invalid_argument("input_signal must have at least 2 dimensions");
    }

    int64_t const dim_h = resolve_dim(dims[0], ndim);
    int64_t const dim_w = resolve_dim(dims[1], ndim);
    if (dim_h == dim_w) {
        throw std::invalid_argument("dims must refer to two different dimensions");
    }

    // Move spatial dims to the last two positions -> [..., H, W]
    auto x = torch::movedim(input_signal,
                            std::vector<int64_t>{dim_h, dim_w},
                            std::vector<int64_t>{ndim - 2, ndim - 1});
    auto const orig_shape = x.sizes().vec();
    int64_t const H = orig_shape[ndim - 2];
    int64_t const W = orig_shape[ndim - 1];

    validate_packet_args({H, W}, max_level, wavelet.dec_len());

    // Flatten leading dims -> [batch, H, W]
    auto flat = x.reshape({-1, H, W});
    auto const opts = torch::TensorOptions().dtype(flat.dtype()).device(flat.device());

    auto subbands = flat;

    std::vector<torch::Tensor> level_outputs;
    level_outputs.reserve(max_level);

    for (int64_t level = 1; level <= max_level; ++level) {
        // At level L each axis has num_subbands_before_split subbands of size M from the previous level.
        int64_t const subband_height = subband_size_before_split(H, level);
        int64_t const subband_width = subband_size_before_split(W, level);
        int64_t const num_subbands_per_axis = num_subbands_before_split(level);

        // Build per-subband analysis matrix, then replicate across all subbands.
        auto A_col = construct_boundary_a(wavelet, subband_width, opts, orth_method);
        auto A_row = construct_boundary_a(wavelet, subband_height, opts, orth_method);
        if (num_subbands_per_axis > 1) {
            A_col = block_diag_repeat(A_col, num_subbands_per_axis);
            A_row = block_diag_repeat(A_row, num_subbands_per_axis);
        }

        // Separable analysis: transform width then height (order doesn't matter,
        // the operations commute since they act on different axes).
        subbands = apply_matrix_along_cols(A_col, subbands);
        subbands = apply_matrix_along_rows(A_row, subbands);

        // Reorder from natural to frequency order for output.
        // Note: subbands itself stays in natural order for the next level.
        int64_t const num_subbands = num_subbands_after_split(level);
        auto perm = natural_to_freq_permutation(level);
        auto freq_ordered = permute_subbands_2d(subbands, num_subbands, perm);
        level_outputs.push_back(freq_ordered);
    }

    // Stack levels -> [batch, max_level, H, W]
    auto coeffs = torch::stack(level_outputs, /*dim=*/1);

    // Unflatten batch dims -> [..., max_level, H, W]
    std::vector<int64_t> result_shape(orig_shape.begin(), orig_shape.end() - 2);
    result_shape.push_back(max_level);
    result_shape.push_back(H);
    result_shape.push_back(W);
    auto result = coeffs.reshape(result_shape);

    // Restore dimension ordering: move [max_level, H, W] back.
    // Level dim goes at min(dim_h, dim_w); H and W shift by +1 because
    // the level dim is inserted before them.
    int64_t const insert_pos = std::min(dim_h, dim_w);
    result = torch::movedim(result,
                            std::vector<int64_t>{ndim - 2, ndim - 1, ndim},
                            std::vector<int64_t>{insert_pos, dim_h + 1, dim_w + 1});

    return result;
}

torch::Tensor wavelet_packet_inverse_2d(
    torch::Tensor const& leaf_coeffs,
    Wavelet const& wavelet,
    std::array<int64_t, 2> dims,
    int64_t max_level,
    OrthMethod orth_method) {

    int64_t const ndim = leaf_coeffs.dim();
    if (ndim < 2) {
        throw std::invalid_argument("leaf_coeffs must have at least 2 dimensions");
    }

    int64_t const dim_h = resolve_dim(dims[0], ndim);
    int64_t const dim_w = resolve_dim(dims[1], ndim);
    if (dim_h == dim_w) {
        throw std::invalid_argument("dims must refer to two different dimensions");
    }

    // Move spatial dims to last two positions -> [..., H, W]
    auto x = torch::movedim(leaf_coeffs,
                            std::vector<int64_t>{dim_h, dim_w},
                            std::vector<int64_t>{ndim - 2, ndim - 1});
    auto const orig_shape = x.sizes().vec();
    int64_t const H = orig_shape[ndim - 2];
    int64_t const W = orig_shape[ndim - 1];

    validate_packet_args({H, W}, max_level, wavelet.dec_len());

    // Flatten leading dims -> [batch, H, W]
    auto flat = x.reshape({-1, H, W});
    auto const opts = torch::TensorOptions().dtype(flat.dtype()).device(flat.device());

    auto subbands = flat;

    // Input is in frequency order; invert the permutation to get natural order.
    // permute_subbands_2d with inv_perm reverses the transposed Gray-code mapping.
    {
        int64_t const num_subbands = num_subbands_after_split(max_level);
        auto perm = natural_to_freq_permutation(max_level);
        auto inv_perm = invert_permutation(perm);
        subbands = permute_subbands_2d(subbands, num_subbands, inv_perm);
    }

    // Inverse pass: synthesize from the deepest level back to level 1.
    for (int64_t level = max_level; level >= 1; --level) {
        int64_t const subband_height = subband_size_before_split(H, level);
        int64_t const subband_width = subband_size_before_split(W, level);
        int64_t const num_subbands_per_axis = num_subbands_before_split(level);

        // Build boundary-orthogonalized synthesis matrices, replicated across all subbands.
        auto S_col = construct_boundary_s(wavelet, subband_width, opts, orth_method);
        auto S_row = construct_boundary_s(wavelet, subband_height, opts, orth_method);
        if (num_subbands_per_axis > 1) {
            S_col = block_diag_repeat(S_col, num_subbands_per_axis);
            S_row = block_diag_repeat(S_row, num_subbands_per_axis);
        }

        // Separable synthesis: undo the column and row transforms.
        subbands = apply_matrix_along_cols(S_col, subbands);
        subbands = apply_matrix_along_rows(S_row, subbands);
    }

    auto result = subbands;  // [batch, H, W]

    // Unflatten batch dims -> [..., H, W]
    std::vector<int64_t> result_shape(orig_shape.begin(), orig_shape.end() - 2);
    result_shape.push_back(H);
    result_shape.push_back(W);
    result = result.reshape(result_shape);

    // Restore dimension ordering: move H, W back to original positions.
    result = torch::movedim(result,
                            std::vector<int64_t>{ndim - 2, ndim - 1},
                            std::vector<int64_t>{dim_h, dim_w});

    return result;
}

torch::Tensor natural_to_freq_permutation(int64_t level) {
    // Gray code ordering matching ptwt's _get_graycode_order.
    // In natural order, 'a'-prefixed subbands occupy indices [0, half),
    // 'd'-prefixed occupy [half, 2*half). Prepending 'a' keeps the index,
    // prepending 'd' offsets by half, and the 'd' half is reversed.
    std::vector<int64_t> order = {0, 1};
    for (int64_t l = 1; l < level; ++l) {
        auto const half = static_cast<int64_t>(order.size());
        std::vector<int64_t> next;
        next.reserve(half * 2);
        for (auto idx : order) {
            next.push_back(idx);            // prepend 'a': same index
        }
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            next.push_back(*it + half);     // prepend 'd': offset, reversed
        }
        order = std::move(next);
    }
    return torch::tensor(order, torch::kLong);
}
