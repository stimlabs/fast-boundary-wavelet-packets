#include "transform.hpp"

#include "matrix_build.hpp"
#include "sparse_math.hpp"
#include "wavelet.hpp"

#include <stdexcept>
#include <string>

/// Permute subbands in a [N, batch] column-major tensor.
/// perm is [num_subbands] giving the order to read subbands from.
static torch::Tensor permute_subbands(
    torch::Tensor const& data,
    int64_t num_subbands,
    torch::Tensor const& perm) {

    int64_t const N = data.size(0);
    int64_t const subband_size = N / num_subbands;
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

static void validate_packet_args(
    int64_t N,
    int64_t& max_level,
    int64_t dec_len) {

    if (max_level == -1) {
        max_level = compute_max_level(N, dec_len);
        if (max_level < 1) {
            throw std::invalid_argument(
                "Cannot auto-compute max_level: signal length " + std::to_string(N) +
                " with filter length " + std::to_string(dec_len) + " yields level 0");
        }
        return;
    }
    if (max_level < 1) {
        throw std::invalid_argument("max_level must be >= 1, got " + std::to_string(max_level));
    }
    int64_t const divisor = 1LL << max_level;
    if (N % divisor != 0) {
        throw std::invalid_argument(
            "Signal length " + std::to_string(N) +
            " is not divisible by 2^max_level = " + std::to_string(divisor));
    }
    // Smallest subband at the deepest split must be >= dec_len
    int64_t const min_subband = N / (1LL << (max_level - 1));
    if (min_subband < dec_len) {
        throw std::invalid_argument(
            "Subband size " + std::to_string(min_subband) +
            " at level " + std::to_string(max_level) +
            " is smaller than filter length " + std::to_string(dec_len));
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

torch::Tensor wavelet_packet_forward_1d(
    torch::Tensor const& input_signal,
    std::string const& wavelet_name,
    int64_t dim,
    int64_t max_level,
    OrthMethod orth_method) {

    auto const w = make_wavelet(wavelet_name);

    // Resolve dim and move analyzed dimension to last position.
    int64_t const ndim = input_signal.dim();
    if (ndim < 1) {
        throw std::invalid_argument("input_signal must have at least 1 dimension");
    }
    int64_t const resolved_dim = resolve_dim(dim, ndim);

    auto x = torch::movedim(input_signal, resolved_dim, -1);  // [..., N]
    auto const orig_shape = x.sizes().vec();                    // shape with N at the end
    int64_t const N = orig_shape.back();

    validate_packet_args(N, max_level, w.dec_len());

    // Flatten leading dims into a single batch dim → [batch, N]
    auto flat = x.reshape({-1, N});

    auto const opts = torch::TensorOptions().dtype(flat.dtype()).device(flat.device());

    // Working state: [N, batch] for efficient sparse mm
    auto subbands = flat.t().contiguous();

    // Collect each level's output
    std::vector<torch::Tensor> level_outputs;
    level_outputs.reserve(max_level);

    for (int64_t level = 1; level <= max_level; ++level) {
        int64_t const M = N / (1LL << (level - 1));  // subband size before split
        int64_t const k = 1LL << (level - 1);        // number of subbands

        auto A_block = construct_boundary_a(w, M, opts, orth_method);
        if (k > 1) {
            A_block = block_diag_repeat(A_block, k);
        }

        subbands = torch::mm(A_block, subbands);

        // Permute from natural to frequency order for output
        int64_t const num_subbands = 1LL << level;
        auto perm = natural_to_freq_permutation(level);
        auto freq_ordered = permute_subbands(subbands, num_subbands, perm);
        level_outputs.push_back(freq_ordered.t().contiguous());  // [batch, N]
    }

    // Stack levels → [batch, max_level, N]
    auto coeffs = torch::stack(level_outputs, /*dim=*/1);

    // Unflatten batch dims back to original shape
    // orig_shape is [..., N], we want [..., max_level, N]
    std::vector<int64_t> result_shape(orig_shape.begin(), orig_shape.end() - 1);
    result_shape.push_back(max_level);
    result_shape.push_back(N);
    auto result = coeffs.reshape(result_shape);  // [..., max_level, N]

    // Move max_level and N dims back to resolved_dim position
    // Currently: [..., max_level, N] where max_level is at ndim-1 and N is at ndim
    // Target: level dim at resolved_dim, analyzed dim at resolved_dim+1
    result = torch::movedim(result, std::vector<int64_t>{ndim - 1, ndim},
                                    std::vector<int64_t>{resolved_dim, resolved_dim + 1});

    return result;
}

torch::Tensor wavelet_packet_inverse_1d(
    torch::Tensor const& leaf_coeffs,
    std::string const& wavelet_name,
    int64_t dim,
    int64_t max_level,
    OrthMethod orth_method) {

    auto const w = make_wavelet(wavelet_name);

    // Resolve dim and move analyzed dimension to last position.
    int64_t const ndim = leaf_coeffs.dim();
    if (ndim < 1) {
        throw std::invalid_argument("leaf_coeffs must have at least 1 dimension");
    }
    int64_t const resolved_dim = resolve_dim(dim, ndim);

    auto x = torch::movedim(leaf_coeffs, resolved_dim, -1);  // [..., N]
    auto const orig_shape = x.sizes().vec();
    int64_t const N = orig_shape.back();

    validate_packet_args(N, max_level, w.dec_len());

    // Flatten leading dims into a single batch dim → [batch, N]
    auto flat = x.reshape({-1, N});

    auto const opts = torch::TensorOptions().dtype(flat.dtype()).device(flat.device());

    // Working state: [N, batch]
    auto subbands = flat.t().contiguous();

    // Input is in frequency order; convert to natural for the first iteration.
    {
        int64_t const num_subbands = 1LL << max_level;
        auto perm = natural_to_freq_permutation(max_level);
        // We need freq→natural: invert the nat→freq permutation.
        auto inv_perm = torch::empty_like(perm);
        inv_perm.scatter_(0, perm, torch::arange(num_subbands, torch::kLong));
        subbands = permute_subbands(subbands, num_subbands, inv_perm);
    }

    for (int64_t level = max_level; level >= 1; --level) {
        int64_t const M = N / (1LL << (level - 1));
        int64_t const k = 1LL << (level - 1);

        auto S_block = construct_boundary_s(w, M, opts, orth_method);
        if (k > 1) {
            S_block = block_diag_repeat(S_block, k);
        }

        subbands = torch::mm(S_block, subbands);
    }

    auto result = subbands.t().contiguous();  // [batch, N]

    // Unflatten batch dims back to original shape
    std::vector<int64_t> result_shape(orig_shape.begin(), orig_shape.end() - 1);
    result_shape.push_back(N);
    result = result.reshape(result_shape);  // [..., N]

    // Move N dim back to resolved_dim position
    result = torch::movedim(result, -1, resolved_dim);

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
