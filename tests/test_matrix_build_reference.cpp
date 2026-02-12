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
};

static void test_case(TestCase const& tc) {
    auto const w = make_wavelet(tc.wavelet);
    auto const opts = torch::TensorOptions().dtype(torch::kFloat64);

    auto our_a = construct_a(w, tc.length, opts);
    auto our_s = construct_s(w, tc.length, opts);

    // Ensure dense for comparison.
    if (our_a.is_sparse()) our_a = our_a.to_dense();
    if (our_s.is_sparse()) our_s = our_s.to_dense();

    auto ref_a = load_tensor(
        data_path(tc.wavelet + "_" + std::to_string(tc.length) + "_a.pt"));
    auto ref_s = load_tensor(
        data_path(tc.wavelet + "_" + std::to_string(tc.length) + "_s.pt"));

    // Check shapes match.
    assert(our_a.sizes() == ref_a.sizes());
    assert(our_s.sizes() == ref_s.sizes());

    auto diff_a = (our_a - ref_a).abs().max().item<double>();
    auto diff_s = (our_s - ref_s).abs().max().item<double>();

    if (diff_a > TOL) {
        std::cerr << "FAIL " << tc.wavelet << " len=" << tc.length
                  << " A max diff = " << diff_a << std::endl;
        std::cerr << "Our A:\n" << our_a << std::endl;
        std::cerr << "Ref A:\n" << ref_a << std::endl;
        assert(false);
    }
    if (diff_s > TOL) {
        std::cerr << "FAIL " << tc.wavelet << " len=" << tc.length
                  << " S max diff = " << diff_s << std::endl;
        std::cerr << "Our S:\n" << our_s << std::endl;
        std::cerr << "Ref S:\n" << ref_s << std::endl;
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
