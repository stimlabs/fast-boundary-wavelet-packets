#include "matrix_build.h"
#include "sparse_math.h"

#include <cassert>
#include <cmath>
#include <iostream>

static constexpr double TOL = 1e-10;

// Verify A has shape (length, length) for various input sizes.
static void test_a_shape() {
    auto const w = make_wavelet("haar");
    auto const opts = torch::TensorOptions().dtype(torch::kFloat64);

    for (int64_t len : {8, 16, 32}) {
        auto a = construct_a(w, len, opts);
        assert(a.size(0) == len);
        assert(a.size(1) == len);
    }
    std::cout << "  test_a_shape passed." << std::endl;
}

// Verify S has shape (length, length) for various input sizes.
static void test_s_shape() {
    auto const w = make_wavelet("haar");
    auto const opts = torch::TensorOptions().dtype(torch::kFloat64);

    for (int64_t len : {8, 16, 32}) {
        auto s = construct_s(w, len, opts);
        assert(s.size(0) == len);
        assert(s.size(1) == len);
    }
    std::cout << "  test_s_shape passed." << std::endl;
}

// For Haar, S @ A = I exactly (filters are length 2, no boundary issues).
static void test_haar_perfect_reconstruction() {
    auto const w = make_wavelet("haar");
    auto const opts = torch::TensorOptions().dtype(torch::kFloat64);
    int64_t const n = 8;

    auto a = construct_a(w, n, opts);
    auto s = construct_s(w, n, opts);
    auto sa = torch::mm(s, a);
    auto eye = torch::eye(n, opts);

    auto diff = (sa - eye).abs().max().item<double>();
    if (diff > TOL) {
        std::cerr << "S @ A not identity, max diff = " << diff << std::endl;
        assert(false);
    }
    std::cout << "  test_haar_perfect_reconstruction passed." << std::endl;
}

// Top half of A matches dec_lo strided conv, bottom half matches dec_hi.
static void test_a_lowpass_highpass_split() {
    auto const w = make_wavelet("haar");
    auto const opts = torch::TensorOptions().dtype(torch::kFloat64);
    int64_t const n = 8;

    auto a = construct_a(w, n, opts);

    auto dec_lo = torch::tensor(w.dec_lo, opts);
    auto dec_hi = torch::tensor(w.dec_hi, opts);
    auto expected_lo = construct_strided_conv_matrix(dec_lo, n, 2);
    auto expected_hi = construct_strided_conv_matrix(dec_hi, n, 2);

    auto top = a.slice(0, 0, n / 2);
    auto bot = a.slice(0, n / 2, n);

    auto diff_lo = (top - expected_lo).abs().max().item<double>();
    auto diff_hi = (bot - expected_hi).abs().max().item<double>();

    if (diff_lo > TOL) {
        std::cerr << "A top half != dec_lo strided conv, max diff = " << diff_lo << std::endl;
        assert(false);
    }
    if (diff_hi > TOL) {
        std::cerr << "A bottom half != dec_hi strided conv, max diff = " << diff_hi << std::endl;
        assert(false);
    }
    std::cout << "  test_a_lowpass_highpass_split passed." << std::endl;
}

int main() {
    test_a_shape();
    test_s_shape();
    test_haar_perfect_reconstruction();
    test_a_lowpass_highpass_split();

    std::cout << "All matrix_build tests passed." << std::endl;
    return 0;
}
