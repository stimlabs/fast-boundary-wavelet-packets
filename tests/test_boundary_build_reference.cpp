#include "matrix_build.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifndef TEST_DATA_DIR
#error "TEST_DATA_DIR must be defined at compile time"
#endif

static constexpr double TOL = 1e-10;

static torch::Tensor load_tensor(std::string const& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << path << std::endl;
        assert(false);
    }
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    auto ivalue = torch::pickle_load(bytes);
    return ivalue.toTensor().to(torch::kFloat64).contiguous();
}

static std::string data_path(std::string const& filename) {
    return std::string(TEST_DATA_DIR) + "/" + filename;
}

struct TestCase {
    std::string wavelet;
    int64_t length;
    OrthMethod method;
    std::string method_tag;  // "qr" or "gs" for filename
};

static void test_case(TestCase const& tc) {
    auto const w = make_wavelet(tc.wavelet);
    auto const opts = torch::TensorOptions().dtype(torch::kFloat64);

    auto our_a = construct_boundary_a(w, tc.length, opts, tc.method);
    auto our_s = construct_boundary_s(w, tc.length, opts, tc.method);

    assert(our_a.is_sparse());
    assert(our_s.is_sparse());

    auto our_a_dense = our_a.to_dense();
    auto our_s_dense = our_s.to_dense();

    auto ref_a = load_tensor(
        data_path(tc.wavelet + "_" + std::to_string(tc.length) + "_boundary_" + tc.method_tag + "_a.pt"));
    auto ref_s = load_tensor(
        data_path(tc.wavelet + "_" + std::to_string(tc.length) + "_boundary_" + tc.method_tag + "_s.pt"));

    assert(our_a_dense.sizes() == ref_a.sizes());
    assert(our_s_dense.sizes() == ref_s.sizes());

    auto diff_a = (our_a_dense - ref_a).abs().max().item<double>();
    auto diff_s = (our_s_dense - ref_s).abs().max().item<double>();

    if (diff_a > TOL) {
        std::cerr << "FAIL " << tc.wavelet << " len=" << tc.length
                  << " method=" << tc.method_tag
                  << " boundary A max diff = " << diff_a << std::endl;
        assert(false);
    }
    if (diff_s > TOL) {
        std::cerr << "FAIL " << tc.wavelet << " len=" << tc.length
                  << " method=" << tc.method_tag
                  << " boundary S max diff = " << diff_s << std::endl;
        assert(false);
    }

    std::cout << "  " << tc.wavelet << " len=" << tc.length
              << " method=" << tc.method_tag
              << " OK (A diff=" << diff_a << ", S diff=" << diff_s << ")"
              << std::endl;
}

int main() {
    std::vector<TestCase> const cases = {
        {"haar", 8, OrthMethod::qr, "qr"},
        {"haar", 16, OrthMethod::qr, "qr"},
        {"db2", 8, OrthMethod::qr, "qr"},
        {"db2", 16, OrthMethod::qr, "qr"},
        {"db3", 16, OrthMethod::qr, "qr"},
        {"haar", 8, OrthMethod::gramschmidt, "gs"},
        {"haar", 16, OrthMethod::gramschmidt, "gs"},
        {"db2", 8, OrthMethod::gramschmidt, "gs"},
        {"db2", 16, OrthMethod::gramschmidt, "gs"},
        {"db3", 16, OrthMethod::gramschmidt, "gs"},
    };

    for (auto const& tc : cases) {
        test_case(tc);
    }

    std::cout << "All boundary build reference tests passed." << std::endl;
    return 0;
}
