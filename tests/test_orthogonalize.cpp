#include "matrix_build.h"
#include "orthogonalize.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

static constexpr double TOL = 1e-10;

static void test_find_boundary_rows_haar() {
    auto const w = make_wavelet("haar");
    auto const opts = torch::TensorOptions().dtype(torch::kFloat64);
    auto a = construct_a(w, 8, opts);

    assert(a.is_sparse());
    auto boundary = find_boundary_rows(a, w.dec_len());
    assert(boundary.numel() == 0);
    std::cout << "  test_find_boundary_rows_haar passed." << std::endl;
}

static void test_find_boundary_rows_db2() {
    auto const w = make_wavelet("db2");
    auto const opts = torch::TensorOptions().dtype(torch::kFloat64);
    auto a = construct_a(w, 8, opts);

    assert(a.is_sparse());
    auto boundary = find_boundary_rows(a, w.dec_len());
    assert(boundary.numel() > 0);
    std::cout << "  test_find_boundary_rows_db2 passed. ("
              << boundary.numel() << " boundary rows)" << std::endl;
}

static void test_orth_preserves_interior_rows(OrthMethod method, std::string const& label) {
    auto const w = make_wavelet("db2");
    auto const opts = torch::TensorOptions().dtype(torch::kFloat64);
    int64_t const n = 16;

    auto a = construct_a(w, n, opts);
    auto boundary = find_boundary_rows(a, w.dec_len());

    torch::Tensor a_orth;
    if (method == OrthMethod::qr) {
        a_orth = orth_by_qr(a, boundary);
    } else {
        a_orth = orth_by_gram_schmidt(a, boundary);
    }

    assert(a_orth.is_sparse());

    auto a_dense = a.to_dense();
    auto a_orth_dense = a_orth.to_dense();

    auto is_boundary = torch::zeros(n, torch::kBool);
    is_boundary.index_fill_(0, boundary, true);

    for (int64_t i = 0; i < n; ++i) {
        if (is_boundary[i].item<bool>()) continue;
        auto diff = (a_orth_dense[i] - a_dense[i]).abs().max().item<double>();
        if (diff > TOL) {
            std::cerr << "Interior row " << i << " changed, max diff = " << diff << std::endl;
            assert(false);
        }
    }
    std::cout << "  test_orth_preserves_interior_rows [" << label << "] passed." << std::endl;
}

static void test_orth_orthonormal_rows(OrthMethod method, std::string const& label) {
    auto const w = make_wavelet("db2");
    auto const opts = torch::TensorOptions().dtype(torch::kFloat64);
    int64_t const n = 16;

    auto a = construct_a(w, n, opts);
    auto a_orth = orthogonalize(a, w.dec_len(), method);

    assert(a_orth.is_sparse());

    auto a_dense = a_orth.to_dense();
    auto aat = torch::mm(a_dense, a_dense.t());
    auto eye = torch::eye(n, opts);

    auto diff = (aat - eye).abs().max().item<double>();
    if (diff > TOL) {
        std::cerr << "A @ A.T not identity, max diff = " << diff << std::endl;
        assert(false);
    }
    std::cout << "  test_orth_orthonormal_rows [" << label << "] passed. (max diff=" << diff << ")" << std::endl;
}

static void test_boundary_perfect_reconstruction(OrthMethod method, std::string const& label) {
    auto const opts = torch::TensorOptions().dtype(torch::kFloat64);

    struct Case { std::string wavelet; int64_t length; };
    std::vector<Case> const cases = {
        {"db2", 8}, {"db2", 16}, {"db3", 16},
    };

    for (auto const& c : cases) {
        auto const w = make_wavelet(c.wavelet);
        auto a = construct_boundary_a(w, c.length, opts, method);
        auto s = construct_boundary_s(w, c.length, opts, method);

        assert(a.is_sparse());
        assert(s.is_sparse());

        auto sa = torch::mm(s.to_dense(), a.to_dense());
        auto eye = torch::eye(c.length, opts);
        auto diff = (sa - eye).abs().max().item<double>();

        if (diff > TOL) {
            std::cerr << "FAIL " << c.wavelet << " len=" << c.length
                      << " [" << label << "] S @ A not identity, max diff = " << diff << std::endl;
            assert(false);
        }
        std::cout << "  " << c.wavelet << " len=" << c.length
                  << " [" << label << "] perfect reconstruction OK (diff=" << diff << ")" << std::endl;
    }
    std::cout << "  test_boundary_perfect_reconstruction [" << label << "] passed." << std::endl;
}

static void test_boundary_haar_noop() {
    auto const w = make_wavelet("haar");
    auto const opts = torch::TensorOptions().dtype(torch::kFloat64);
    int64_t const n = 8;

    auto a = construct_a(w, n, opts);
    auto a_boundary = construct_boundary_a(w, n, opts);

    assert(a.is_sparse());
    assert(a_boundary.is_sparse());

    auto diff = (a_boundary.to_dense() - a.to_dense()).abs().max().item<double>();
    if (diff > TOL) {
        std::cerr << "Haar boundary A differs from raw A, max diff = " << diff << std::endl;
        assert(false);
    }
    std::cout << "  test_boundary_haar_noop passed." << std::endl;
}

int main() {
    test_find_boundary_rows_haar();
    test_find_boundary_rows_db2();

    // QR tests
    test_orth_preserves_interior_rows(OrthMethod::qr, "qr");
    test_orth_orthonormal_rows(OrthMethod::qr, "qr");
    test_boundary_perfect_reconstruction(OrthMethod::qr, "qr");

    // Gram-Schmidt tests
    test_orth_preserves_interior_rows(OrthMethod::gramschmidt, "gs");
    test_orth_orthonormal_rows(OrthMethod::gramschmidt, "gs");
    test_boundary_perfect_reconstruction(OrthMethod::gramschmidt, "gs");

    test_boundary_haar_noop();

    std::cout << "All orthogonalize tests passed." << std::endl;
    return 0;
}
