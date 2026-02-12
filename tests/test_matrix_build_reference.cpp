#include "matrix_build.hpp"
#include "test_util.hpp"

#include <iostream>
#include <string>
#include <vector>

static constexpr double TOL = 1e-10;

struct TestCase {
    std::string wavelet;
    int64_t length;
};

static void test_case(TestCase const& tc) {
    auto const w = make_wavelet(tc.wavelet);
    auto const opts = torch::TensorOptions().dtype(torch::kFloat64);

    auto our_a = construct_a(w, tc.length, opts);
    auto our_s = construct_s(w, tc.length, opts);

    assert(our_a.is_sparse());
    assert(our_s.is_sparse());

    auto our_a_dense = our_a.to_dense();
    auto our_s_dense = our_s.to_dense();

    auto ref_a = load_tensor(
        data_path(tc.wavelet + "_" + std::to_string(tc.length) + "_a.pt"));
    auto ref_s = load_tensor(
        data_path(tc.wavelet + "_" + std::to_string(tc.length) + "_s.pt"));

    assert(our_a_dense.sizes() == ref_a.sizes());
    assert(our_s_dense.sizes() == ref_s.sizes());

    auto diff_a = (our_a_dense - ref_a).abs().max().item<double>();
    auto diff_s = (our_s_dense - ref_s).abs().max().item<double>();

    if (diff_a > TOL) {
        std::cerr << "FAIL " << tc.wavelet << " len=" << tc.length
                  << " A max diff = " << diff_a << std::endl;
        assert(false);
    }
    if (diff_s > TOL) {
        std::cerr << "FAIL " << tc.wavelet << " len=" << tc.length
                  << " S max diff = " << diff_s << std::endl;
        assert(false);
    }

    std::cout << "  " << tc.wavelet << " len=" << tc.length
              << " OK (A diff=" << diff_a << ", S diff=" << diff_s << ")"
              << std::endl;
}

int main() {
    std::vector<TestCase> const cases = {
        {"haar", 8},
        {"haar", 16},
        {"db2", 8},
        {"db2", 16},
        {"db3", 16},
    };

    for (auto const& tc : cases) {
        test_case(tc);
    }

    std::cout << "All reference match tests passed." << std::endl;
    return 0;
}
