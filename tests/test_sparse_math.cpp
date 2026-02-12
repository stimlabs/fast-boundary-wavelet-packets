#include "sparse_math.hpp"
#include "wavelet.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

static constexpr double TOL = 1e-10;

static void assert_near(double const a, double const b, char const* msg) {
    if (std::abs(a - b) > TOL) {
        std::cerr << msg << ": expected " << b << ", got " << a << std::endl;
        assert(false);
    }
}

// filter [1,2,3,4], input_length=8 → dense 8x8 with filter coefficients at
// expected positions (sameshift, start_row=1).
static void test_conv_matrix_structure() {
    auto filter = torch::tensor({1.0, 2.0, 3.0, 4.0}, torch::kFloat64);
    auto sparse = construct_conv_matrix(filter, 8);
    auto dense = sparse.to_dense();

    assert(dense.size(0) == 8);
    assert(dense.size(1) == 8);

    // Expected dense matrix (sameshift with filter_length=4, start_row=1):
    //   row 0: [2, 1, 0, 0, 0, 0, 0, 0]
    //   row 1: [3, 2, 1, 0, 0, 0, 0, 0]
    //   row 2: [4, 3, 2, 1, 0, 0, 0, 0]
    //   row 3: [0, 4, 3, 2, 1, 0, 0, 0]
    //   row 4: [0, 0, 4, 3, 2, 1, 0, 0]
    //   row 5: [0, 0, 0, 4, 3, 2, 1, 0]
    //   row 6: [0, 0, 0, 0, 4, 3, 2, 1]
    //   row 7: [0, 0, 0, 0, 0, 4, 3, 2]
    double expected[8][8] = {
        {2, 1, 0, 0, 0, 0, 0, 0},
        {3, 2, 1, 0, 0, 0, 0, 0},
        {4, 3, 2, 1, 0, 0, 0, 0},
        {0, 4, 3, 2, 1, 0, 0, 0},
        {0, 0, 4, 3, 2, 1, 0, 0},
        {0, 0, 0, 4, 3, 2, 1, 0},
        {0, 0, 0, 0, 4, 3, 2, 1},
        {0, 0, 0, 0, 0, 4, 3, 2},
    };

    auto acc = dense.accessor<double, 2>();
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            auto msg = "matrix[" + std::to_string(r) + "][" + std::to_string(c) + "]";
            assert_near(acc[r][c], expected[r][c], msg.c_str());
        }
    }
    std::cout << "  test_conv_matrix_structure passed." << std::endl;
}

// Output is always (input_length x input_length) for various filter/input sizes.
static void test_conv_matrix_square() {
    int64_t const filter_lengths[] = {2, 3, 4, 6, 8};
    int64_t const input_lengths[] = {4, 8, 16, 32};

    for (auto fl : filter_lengths) {
        auto filter = torch::randn({fl}, torch::kFloat64);
        for (auto il : input_lengths) {
            auto sparse = construct_conv_matrix(filter, il);
            assert(sparse.size(0) == il);
            assert(sparse.size(1) == il);
        }
    }
    std::cout << "  test_conv_matrix_square passed." << std::endl;
}

// Strided version with stride=2 selects rows 1, 3, 5, ... of the full matrix.
static void test_strided_conv_matrix_selects_odd_rows() {
    auto filter = torch::tensor({1.0, 2.0, 3.0, 4.0}, torch::kFloat64);
    int64_t const n = 8;

    auto full = construct_conv_matrix(filter, n).to_dense();
    auto strided = construct_strided_conv_matrix(filter, n, 2).to_dense();

    // Should have n/2 = 4 rows
    assert(strided.size(0) == n / 2);
    assert(strided.size(1) == n);

    auto full_acc = full.accessor<double, 2>();
    auto str_acc = strided.accessor<double, 2>();

    for (int64_t i = 0; i < n / 2; ++i) {
        int64_t const full_row = 1 + i * 2;  // rows 1, 3, 5, 7
        for (int64_t c = 0; c < n; ++c) {
            auto msg = "strided[" + std::to_string(i) + "][" + std::to_string(c) + "]"
                       + " vs full[" + std::to_string(full_row) + "][" + std::to_string(c) + "]";
            assert_near(str_acc[i][c], full_acc[full_row][c], msg.c_str());
        }
    }
    std::cout << "  test_strided_conv_matrix_selects_odd_rows passed." << std::endl;
}

// Haar analysis matrix (dec_lo and dec_hi stacked, stride=2) is orthogonal: A @ A^T = I.
static void test_strided_haar_orthogonality() {
    auto const w = make_wavelet("haar");
    int64_t const n = 8;

    auto lo = torch::tensor(w.dec_lo, torch::kFloat64);
    auto hi = torch::tensor(w.dec_hi, torch::kFloat64);

    auto a_lo = construct_strided_conv_matrix(lo, n, 2).to_dense();
    auto a_hi = construct_strided_conv_matrix(hi, n, 2).to_dense();

    // Stack vertically: A is (n, n)
    auto a = torch::cat({a_lo, a_hi}, /*dim=*/0);
    assert(a.size(0) == n);
    assert(a.size(1) == n);

    auto aat = torch::mm(a, a.t());
    auto eye = torch::eye(n, torch::kFloat64);
    auto diff = (aat - eye).abs().max().item<double>();

    if (diff > TOL) {
        std::cerr << "A @ A^T not identity, max diff = " << diff << std::endl;
        assert(false);
    }
    std::cout << "  test_strided_haar_orthogonality passed." << std::endl;
}

int main() {
    test_conv_matrix_structure();
    test_conv_matrix_square();
    test_strided_conv_matrix_selects_odd_rows();
    test_strided_haar_orthogonality();

    std::cout << "All sparse_math tests passed." << std::endl;
    return 0;
}
