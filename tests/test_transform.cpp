#include "transform.hpp"
#include "sparse_math.hpp"
#include "wavelet.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

static constexpr double TOL = 1e-7;

// --- block_diag_repeat tests ---

static void test_block_diag_repeat_identity() {
    auto const opts = torch::TensorOptions().dtype(torch::kFloat64);
    int64_t const M = 4;
    auto eye = torch::eye(M, opts).to_sparse();

    auto result = block_diag_repeat(eye, 3);
    assert(result.is_sparse());
    assert(result.size(0) == 12);
    assert(result.size(1) == 12);

    auto dense = result.to_dense();
    auto expected = torch::eye(12, opts);
    auto diff = (dense - expected).abs().max().item<double>();
    assert(diff < TOL);
    std::cout << "  test_block_diag_repeat_identity passed." << std::endl;
}

static void test_block_diag_repeat_values() {
    auto const opts = torch::TensorOptions().dtype(torch::kFloat64);
    // 2x2 block: [[1, 2], [3, 4]]
    auto block = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, opts).to_sparse();

    auto result = block_diag_repeat(block, 2).to_dense();
    auto expected = torch::tensor({
        {1.0, 2.0, 0.0, 0.0},
        {3.0, 4.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 2.0},
        {0.0, 0.0, 3.0, 4.0}}, opts);

    auto diff = (result - expected).abs().max().item<double>();
    assert(diff < TOL);
    std::cout << "  test_block_diag_repeat_values passed." << std::endl;
}

static void test_block_diag_repeat_single() {
    auto const opts = torch::TensorOptions().dtype(torch::kFloat64);
    auto block = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, opts).to_sparse();

    auto result = block_diag_repeat(block, 1).to_dense();
    auto expected = block.to_dense();

    auto diff = (result - expected).abs().max().item<double>();
    assert(diff < TOL);
    std::cout << "  test_block_diag_repeat_single passed." << std::endl;
}

// --- compute_max_level tests ---

static void test_compute_max_level() {
    // haar: dec_len = 2
    assert(compute_max_level(8, 2) == 3);   // 8/1>=2, 8/2>=2, 8/4>=2, 8/8<2 → L=3
    assert(compute_max_level(16, 2) == 4);
    assert(compute_max_level(3, 2) == 0);    // 3 is odd → L=0

    // db2: dec_len = 4
    assert(compute_max_level(16, 4) == 3);   // 16/1>=4, 16/2>=4, 16/4>=4, 16/8<4 → L=3
    assert(compute_max_level(8, 4) == 2);    // 8/1>=4, 8/2>=4, 8/4<4 → L=2

    // db3: dec_len = 6
    assert(compute_max_level(16, 6) == 2);   // 16/1>=6, 16/2>=6, 16/4<6 → L=2
    assert(compute_max_level(32, 6) == 3);   // 32/1>=6, 32/2>=6, 32/4>=6, 32/8<6 → L=3

    std::cout << "  test_compute_max_level passed." << std::endl;
}

// --- Output shape tests ---

static void test_shapes_1d() {
    int64_t const N = 16;
    int64_t const L = 3;

    // Unbatched: [N] → [L, N]
    {
        auto signal = torch::randn({N}, torch::kFloat64);

        auto fwd = wavelet_packet_forward_1d(signal, make_wavelet("haar"), -1, L);
        assert(fwd.dim() == 2);
        assert(fwd.size(0) == L);
        assert(fwd.size(1) == N);

        auto leaf = fwd.select(0, L - 1);  // [N]
        assert(leaf.dim() == 1);
        auto rec = wavelet_packet_inverse_1d(leaf, make_wavelet("haar"), -1, L);
        assert(rec.dim() == 1);
        assert(rec.size(0) == N);
    }

    // Batched: [batch, N] → [batch, L, N]
    {
        int64_t const B = 5;
        auto signal = torch::randn({B, N}, torch::kFloat64);

        auto fwd = wavelet_packet_forward_1d(signal, make_wavelet("haar"), -1, L);
        assert(fwd.dim() == 3);
        assert(fwd.size(0) == B);
        assert(fwd.size(1) == L);
        assert(fwd.size(2) == N);

        auto leaf = fwd.select(1, L - 1);  // [batch, N]
        assert(leaf.dim() == 2);
        assert(leaf.size(0) == B);
        auto rec = wavelet_packet_inverse_1d(leaf, make_wavelet("haar"), -1, L);
        assert(rec.dim() == 2);
        assert(rec.size(0) == B);
        assert(rec.size(1) == N);
    }

    // Multi-batch: [B1, B2, N] → [B1, B2, L, N]
    {
        int64_t const B1 = 3, B2 = 4;
        auto signal = torch::randn({B1, B2, N}, torch::kFloat64);

        auto fwd = wavelet_packet_forward_1d(signal, make_wavelet("haar"), -1, L);
        assert(fwd.dim() == 4);
        assert(fwd.size(0) == B1);
        assert(fwd.size(1) == B2);
        assert(fwd.size(2) == L);
        assert(fwd.size(3) == N);

        auto leaf = fwd.select(2, L - 1);  // [B1, B2, N]
        assert(leaf.dim() == 3);
        auto rec = wavelet_packet_inverse_1d(leaf, make_wavelet("haar"), -1, L);
        assert(rec.dim() == 3);
        assert(rec.sizes() == signal.sizes());

        auto diff = (rec - signal).abs().max().item<double>();
        assert(diff < TOL);
    }

    std::cout << "  test_shapes_1d passed." << std::endl;
}

// --- Auto max_level tests ---

static void test_auto_max_level() {
    int64_t const N = 16;

    // Auto-compute should give L=4 for haar with N=16
    auto signal = torch::randn({N}, torch::kFloat64);
    auto fwd = wavelet_packet_forward_1d(signal, make_wavelet("haar"));
    assert(fwd.dim() == 2);
    assert(fwd.size(0) == 4);  // compute_max_level(16, 2) == 4
    assert(fwd.size(1) == N);

    // Inverse with auto max_level
    auto leaf = fwd.select(0, 3);  // leaf level
    auto rec = wavelet_packet_inverse_1d(leaf, make_wavelet("haar"));
    auto diff = (rec - signal).abs().max().item<double>();
    assert(diff < TOL);

    std::cout << "  test_auto_max_level passed." << std::endl;
}

// --- Dim selection tests ---

static void test_dim_selection_2d() {
    // [N, C] with dim=0: transform along N → [L, N, C]
    int64_t const N = 16, C = 3, L = 3;

    auto signal = torch::randn({N, C}, torch::kFloat64);

    auto fwd = wavelet_packet_forward_1d(signal, make_wavelet("haar"), 0, L);
    assert(fwd.dim() == 3);
    assert(fwd.size(0) == L);
    assert(fwd.size(1) == N);
    assert(fwd.size(2) == C);

    // Each channel should match independent 1-D transform
    for (int64_t c = 0; c < C; ++c) {
        auto signal_c = signal.select(1, c);  // [N]
        auto fwd_c = wavelet_packet_forward_1d(signal_c, make_wavelet("haar"), -1, L);  // [L, N]
        auto fwd_slice = fwd.select(2, c);  // [L, N]
        auto diff = (fwd_slice - fwd_c).abs().max().item<double>();
        assert(diff < TOL);
    }

    // Roundtrip
    auto leaf = fwd.select(0, L - 1);  // [N, C]
    auto rec = wavelet_packet_inverse_1d(leaf, make_wavelet("haar"), 0, L);
    assert(rec.sizes() == signal.sizes());
    auto diff = (rec - signal).abs().max().item<double>();
    assert(diff < TOL);

    std::cout << "  test_dim_selection_2d passed." << std::endl;
}

static void test_dim_selection_3d() {
    // [B, N, C] with dim=1: transform along middle dim → [B, L, N, C]
    int64_t const B = 2, N = 16, C = 3, L = 2;

    auto signal = torch::randn({B, N, C}, torch::kFloat64);

    auto fwd = wavelet_packet_forward_1d(signal, make_wavelet("db2"), 1, L);
    assert(fwd.dim() == 4);
    assert(fwd.size(0) == B);
    assert(fwd.size(1) == L);
    assert(fwd.size(2) == N);
    assert(fwd.size(3) == C);

    // Each (b, c) slice should match independent 1-D transform
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t c = 0; c < C; ++c) {
            auto signal_bc = signal[b].select(1, c);  // [N]
            auto fwd_bc = wavelet_packet_forward_1d(signal_bc, make_wavelet("db2"), -1, L);  // [L, N]
            auto fwd_slice = fwd[b].select(2, c);  // [L, N]
            auto diff = (fwd_slice - fwd_bc).abs().max().item<double>();
            assert(diff < TOL);
        }
    }

    // Roundtrip
    auto leaf = fwd.select(1, L - 1);  // [B, N, C]
    auto rec = wavelet_packet_inverse_1d(leaf, make_wavelet("db2"), 1, L);
    assert(rec.sizes() == signal.sizes());
    auto diff = (rec - signal).abs().max().item<double>();
    assert(diff < TOL);

    std::cout << "  test_dim_selection_3d passed." << std::endl;
}

static void test_dim_selection_negative() {
    // Verify negative dim works: dim=-2 on [B, N, C] should be equivalent to dim=1.
    int64_t const B = 2, N = 8, C = 4, L = 2;

    auto signal = torch::randn({B, N, C}, torch::kFloat64);

    auto fwd_pos = wavelet_packet_forward_1d(signal, make_wavelet("haar"), 1, L);
    auto fwd_neg = wavelet_packet_forward_1d(signal, make_wavelet("haar"), -2, L);
    assert(fwd_pos.sizes() == fwd_neg.sizes());
    auto diff = (fwd_pos - fwd_neg).abs().max().item<double>();
    assert(diff < TOL);

    // Same for inverse
    auto leaf_pos = fwd_pos.select(1, L - 1);
    auto leaf_neg = fwd_neg.select(1, L - 1);
    auto rec_pos = wavelet_packet_inverse_1d(leaf_pos, make_wavelet("haar"), 1, L);
    auto rec_neg = wavelet_packet_inverse_1d(leaf_neg, make_wavelet("haar"), -2, L);
    auto diff_inv = (rec_pos - rec_neg).abs().max().item<double>();
    assert(diff_inv < TOL);

    std::cout << "  test_dim_selection_negative passed." << std::endl;
}

static void test_dim_default_is_last() {
    // dim=-1 (default) on [B, N] should match explicit dim=1.
    int64_t const B = 3, N = 16, L = 3;

    auto signal = torch::randn({B, N}, torch::kFloat64);

    auto fwd_default = wavelet_packet_forward_1d(signal, make_wavelet("haar"), -1, L);
    auto fwd_explicit = wavelet_packet_forward_1d(signal, make_wavelet("haar"), 1, L);
    assert(fwd_default.sizes() == fwd_explicit.sizes());
    auto diff = (fwd_default - fwd_explicit).abs().max().item<double>();
    assert(diff < TOL);

    std::cout << "  test_dim_default_is_last passed." << std::endl;
}

// --- Perfect reconstruction tests ---

static void test_perfect_reconstruction(
    Wavelet const& wavelet,
    int64_t N,
    int64_t max_level,
    OrthMethod method,
    std::string const& label) {

    auto signal = torch::randn({N}, torch::kFloat64);

    auto fwd = wavelet_packet_forward_1d(signal, wavelet, -1, max_level, method);
    auto leaf = fwd.select(0, max_level - 1);  // [N]
    auto reconstructed = wavelet_packet_inverse_1d(leaf, wavelet, -1, max_level, method);

    auto diff = (reconstructed - signal).abs().max().item<double>();
    if (diff > TOL) {
        std::cerr << "FAIL perfect reconstruction " << label
                  << " max diff = " << diff << std::endl;
        assert(false);
    }
    std::cout << "  perfect_reconstruction " << label
              << " OK (diff=" << diff << ")" << std::endl;
}

static void test_perfect_reconstruction_batch() {
    auto signal = torch::randn({5, 16}, torch::kFloat64);

    auto fwd = wavelet_packet_forward_1d(signal, make_wavelet("db2"), -1, 2);
    auto leaf = fwd.select(1, 1);  // [batch, N] — level index 1 is max_level-1 for L=2
    auto reconstructed = wavelet_packet_inverse_1d(leaf, make_wavelet("db2"), -1, 2);

    assert(reconstructed.sizes() == signal.sizes());
    auto diff = (reconstructed - signal).abs().max().item<double>();
    if (diff > TOL) {
        std::cerr << "FAIL batch reconstruction max diff = " << diff << std::endl;
        assert(false);
    }
    std::cout << "  test_perfect_reconstruction_batch passed. (diff=" << diff << ")" << std::endl;
}

// --- Energy preservation test ---

static void test_energy_preservation() {
    auto signal = torch::randn({16}, torch::kFloat64);
    double const signal_energy = signal.norm().item<double>();

    int64_t const L = 3;
    auto fwd = wavelet_packet_forward_1d(signal, make_wavelet("haar"), -1, L);
    // Each level's coefficients should preserve energy (Parseval's theorem)
    for (int64_t l = 0; l < L; ++l) {
        auto level_coeffs = fwd.select(0, l);  // [N]
        double const coeff_energy = level_coeffs.norm().item<double>();
        double const rel_diff = std::abs(coeff_energy - signal_energy) / signal_energy;
        if (rel_diff > TOL) {
            std::cerr << "FAIL energy preservation at level " << (l + 1)
                      << " rel_diff = " << rel_diff << std::endl;
            assert(false);
        }
    }
    std::cout << "  test_energy_preservation passed." << std::endl;
}

// --- Invalid input tests ---

static void test_invalid_max_level_zero() {
    auto signal = torch::randn({8}, torch::kFloat64);
    bool caught = false;
    try {
        wavelet_packet_forward_1d(signal, make_wavelet("haar"), -1, 0);
    } catch (std::invalid_argument const&) {
        caught = true;
    }
    assert(caught);
    std::cout << "  test_invalid_max_level_zero passed." << std::endl;
}

static void test_invalid_non_divisible() {
    auto signal = torch::randn({12}, torch::kFloat64);
    bool caught = false;
    try {
        wavelet_packet_forward_1d(signal, make_wavelet("haar"), -1, 4);  // 12 % 16 != 0
    } catch (std::invalid_argument const&) {
        caught = true;
    }
    assert(caught);
    std::cout << "  test_invalid_non_divisible passed." << std::endl;
}

static void test_invalid_signal_too_short() {
    auto signal = torch::randn({8}, torch::kFloat64);
    bool caught = false;
    try {
        wavelet_packet_forward_1d(signal, make_wavelet("db3"), -1, 2);  // dec_len=6, subband size = 8/2 = 4 < 6
    } catch (std::invalid_argument const&) {
        caught = true;
    }
    assert(caught);
    std::cout << "  test_invalid_signal_too_short passed." << std::endl;
}

// --- Natural to freq permutation tests ---

static void test_natural_to_freq_level1() {
    auto perm = natural_to_freq_permutation(1);
    auto expected = torch::tensor({0, 1}, torch::kLong);
    assert(perm.equal(expected));
    std::cout << "  test_natural_to_freq_level1 passed." << std::endl;
}

static void test_natural_to_freq_level2() {
    // Gray code order for level 2: aa(0), ad(1), dd(3), da(2)
    auto perm = natural_to_freq_permutation(2);
    auto expected = torch::tensor({0, 1, 3, 2}, torch::kLong);
    assert(perm.equal(expected));
    std::cout << "  test_natural_to_freq_level2 passed." << std::endl;
}

static void test_natural_to_freq_level3() {
    // Gray code order for level 3:
    // aaa(0), aad(1), add(3), ada(2), dda(6), ddd(7), dad(5), daa(4)
    auto perm = natural_to_freq_permutation(3);
    auto expected = torch::tensor({0, 1, 3, 2, 6, 7, 5, 4}, torch::kLong);
    assert(perm.equal(expected));
    std::cout << "  test_natural_to_freq_level3 passed." << std::endl;
}

int main() {
    std::cout << "block_diag_repeat tests:" << std::endl;
    test_block_diag_repeat_identity();
    test_block_diag_repeat_values();
    test_block_diag_repeat_single();

    std::cout << "compute_max_level tests:" << std::endl;
    test_compute_max_level();

    std::cout << "Output shape tests:" << std::endl;
    test_shapes_1d();

    std::cout << "Auto max_level tests:" << std::endl;
    test_auto_max_level();

    std::cout << "Dim selection tests:" << std::endl;
    test_dim_selection_2d();
    test_dim_selection_3d();
    test_dim_selection_negative();
    test_dim_default_is_last();

    std::cout << "Perfect reconstruction tests:" << std::endl;
    struct ReconCase {
        std::string wavelet_name;
        int64_t N;
        int64_t max_level;
    };
    std::vector<ReconCase> const recon_cases = {
        {"haar", 8, 3},
        {"haar", 16, 4},
        {"db2", 16, 2},
        {"db2", 16, 3},
        {"db3", 16, 2},
        {"db3", 32, 3},
    };
    for (auto const& rc : recon_cases) {
        auto const w = make_wavelet(rc.wavelet_name);
        for (auto method : {OrthMethod::qr, OrthMethod::gramschmidt}) {
            std::string const method_tag = (method == OrthMethod::qr) ? "qr" : "gs";
            std::string const label = rc.wavelet_name + " N=" + std::to_string(rc.N) +
                " L=" + std::to_string(rc.max_level) + " " + method_tag;
            test_perfect_reconstruction(w, rc.N, rc.max_level, method, label);
        }
    }

    test_perfect_reconstruction_batch();

    std::cout << "Energy preservation tests:" << std::endl;
    test_energy_preservation();

    std::cout << "Invalid input tests:" << std::endl;
    test_invalid_max_level_zero();
    test_invalid_non_divisible();
    test_invalid_signal_too_short();

    std::cout << "Natural to freq permutation tests:" << std::endl;
    test_natural_to_freq_level1();
    test_natural_to_freq_level2();
    test_natural_to_freq_level3();

    std::cout << "All transform tests passed." << std::endl;
    return 0;
}
