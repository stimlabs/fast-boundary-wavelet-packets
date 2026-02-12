#include "transform.hpp"
#include "sparse_math.hpp"

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

// --- Output shape tests ---

static void test_shapes_1d() {
    auto const w = make_wavelet("haar");
    int64_t const N = 16;
    int64_t const L = 3;

    // Unbatched: [N]
    {
        auto signal = torch::randn({N}, torch::kFloat64);

        auto fwd = wavelet_packet_forward_1d(signal, w, L);
        assert(fwd.coeffs.dim() == 2);
        assert(fwd.coeffs.size(0) == L);
        assert(fwd.coeffs.size(1) == N);
        assert(fwd.max_level == L);
        assert(fwd.signal_length == N);

        auto rec = wavelet_packet_inverse_1d(fwd, w);
        assert(rec.dim() == 1);
        assert(rec.size(0) == N);

        auto leaf = fwd.coeffs.select(0, L - 1);
        assert(leaf.dim() == 1);
        auto rec2 = wavelet_packet_inverse_1d(leaf, w, L);
        assert(rec2.dim() == 1);
        assert(rec2.size(0) == N);
    }

    // Batched: [batch, N]
    {
        int64_t const B = 5;
        auto signal = torch::randn({B, N}, torch::kFloat64);

        auto fwd = wavelet_packet_forward_1d(signal, w, L);
        assert(fwd.coeffs.dim() == 3);
        assert(fwd.coeffs.size(0) == B);
        assert(fwd.coeffs.size(1) == L);
        assert(fwd.coeffs.size(2) == N);
        assert(fwd.max_level == L);
        assert(fwd.signal_length == N);

        auto rec = wavelet_packet_inverse_1d(fwd, w);
        assert(rec.dim() == 2);
        assert(rec.size(0) == B);
        assert(rec.size(1) == N);

        auto leaf = fwd.coeffs.select(1, L - 1);
        assert(leaf.dim() == 2);
        assert(leaf.size(0) == B);
        auto rec2 = wavelet_packet_inverse_1d(leaf, w, L);
        assert(rec2.dim() == 2);
        assert(rec2.size(0) == B);
        assert(rec2.size(1) == N);
    }

    std::cout << "  test_shapes_1d passed." << std::endl;
}

// --- Perfect reconstruction tests ---

static void test_perfect_reconstruction(
    std::string const& wavelet_name,
    int64_t N,
    int64_t max_level,
    OrthMethod method,
    std::string const& label) {

    auto const w = make_wavelet(wavelet_name);
    auto signal = torch::randn({N}, torch::kFloat64);

    auto fwd = wavelet_packet_forward_1d(signal, w, max_level, method);
    auto reconstructed = wavelet_packet_inverse_1d(fwd, w, method);

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
    auto const w = make_wavelet("db2");
    auto signal = torch::randn({5, 16}, torch::kFloat64);

    auto fwd = wavelet_packet_forward_1d(signal, w, 2);
    auto reconstructed = wavelet_packet_inverse_1d(fwd, w);

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
    auto const w = make_wavelet("haar");
    auto signal = torch::randn({16}, torch::kFloat64);
    double const signal_energy = signal.norm().item<double>();

    auto fwd = wavelet_packet_forward_1d(signal, w, 3);
    // Each level's coefficients should preserve energy (Parseval's theorem)
    for (int64_t l = 0; l < fwd.max_level; ++l) {
        auto level_coeffs = fwd.coeffs.select(0, l);  // [N]
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
    auto const w = make_wavelet("haar");
    auto signal = torch::randn({8}, torch::kFloat64);
    bool caught = false;
    try {
        wavelet_packet_forward_1d(signal, w, 0);
    } catch (std::invalid_argument const&) {
        caught = true;
    }
    assert(caught);
    std::cout << "  test_invalid_max_level_zero passed." << std::endl;
}

static void test_invalid_non_divisible() {
    auto const w = make_wavelet("haar");
    auto signal = torch::randn({12}, torch::kFloat64);
    bool caught = false;
    try {
        wavelet_packet_forward_1d(signal, w, 4);  // 12 % 16 != 0
    } catch (std::invalid_argument const&) {
        caught = true;
    }
    assert(caught);
    std::cout << "  test_invalid_non_divisible passed." << std::endl;
}

static void test_invalid_signal_too_short() {
    auto const w = make_wavelet("db3");  // dec_len = 6
    auto signal = torch::randn({8}, torch::kFloat64);
    bool caught = false;
    try {
        wavelet_packet_forward_1d(signal, w, 2);  // subband size = 8/2 = 4 < 6
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

    std::cout << "Output shape tests:" << std::endl;
    test_shapes_1d();

    std::cout << "Perfect reconstruction tests:" << std::endl;
    struct ReconCase {
        std::string wavelet;
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
        for (auto method : {OrthMethod::qr, OrthMethod::gramschmidt}) {
            std::string const method_tag = (method == OrthMethod::qr) ? "qr" : "gs";
            std::string const label = rc.wavelet + " N=" + std::to_string(rc.N) +
                " L=" + std::to_string(rc.max_level) + " " + method_tag;
            test_perfect_reconstruction(rc.wavelet, rc.N, rc.max_level, method, label);
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
