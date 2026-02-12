#include "transform.hpp"

#include "matrix_build.hpp"
#include "sparse_math.hpp"

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

static void validate_packet_args(
    int64_t N,
    int64_t max_level,
    int64_t dec_len) {

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

WaveletPacketResult wavelet_packet_forward_1d(
    torch::Tensor const& signal,
    Wavelet const& wavelet,
    int64_t max_level,
    OrthMethod method) {

    bool const was_1d = signal.dim() == 1;
    auto x = was_1d ? signal.unsqueeze(0) : signal;  // [batch, N]
    int64_t const N = x.size(1);

    validate_packet_args(N, max_level, wavelet.dec_len());

    auto const opts = torch::TensorOptions().dtype(x.dtype()).device(x.device());

    // Working state: [N, batch] for efficient sparse mm
    auto subbands = x.t().contiguous();

    // Collect each level's output
    std::vector<torch::Tensor> level_outputs;
    level_outputs.reserve(max_level);

    for (int64_t level = 1; level <= max_level; ++level) {
        int64_t const M = N / (1LL << (level - 1));  // subband size before split
        int64_t const k = 1LL << (level - 1);        // number of subbands

        auto A_block = construct_boundary_a(wavelet, M, opts, method);
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

    if (was_1d) {
        coeffs = coeffs.squeeze(0);  // [max_level, N]
    }

    return WaveletPacketResult{coeffs, max_level, N};
}

torch::Tensor wavelet_packet_inverse_1d(
    torch::Tensor const& leaf_coeffs,
    Wavelet const& wavelet,
    int64_t max_level,
    OrthMethod method) {

    bool const was_1d = leaf_coeffs.dim() == 1;
    auto x = was_1d ? leaf_coeffs.unsqueeze(0) : leaf_coeffs;  // [batch, N]
    int64_t const N = x.size(1);

    validate_packet_args(N, max_level, wavelet.dec_len());

    auto const opts = torch::TensorOptions().dtype(x.dtype()).device(x.device());

    // Working state: [N, batch]
    auto subbands = x.t().contiguous();

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

        auto S_block = construct_boundary_s(wavelet, M, opts, method);
        if (k > 1) {
            S_block = block_diag_repeat(S_block, k);
        }

        subbands = torch::mm(S_block, subbands);
    }

    auto result = subbands.t().contiguous();  // [batch, N]

    if (was_1d) {
        result = result.squeeze(0);  // [N]
    }
    return result;
}

torch::Tensor wavelet_packet_inverse_1d(
    WaveletPacketResult const& result,
    Wavelet const& wavelet,
    OrthMethod method) {

    // Extract leaf level (last level)
    auto leaf = result.coeffs;
    if (leaf.dim() == 3) {
        // [batch, max_level, N] → take last level
        leaf = leaf.select(1, result.max_level - 1);  // [batch, N]
    } else {
        // [max_level, N] → take last level
        leaf = leaf.select(0, result.max_level - 1);  // [N]
    }
    return wavelet_packet_inverse_1d(leaf, wavelet, result.max_level, method);
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
