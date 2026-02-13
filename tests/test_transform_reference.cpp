#include "transform.hpp"
#include "test_util.hpp"

#include <iostream>
#include <string>
#include <vector>

static constexpr double TOL = 1e-7;

struct WPTestCase {
    std::string dimension;
    std::string wavelet;
    int64_t length;
    int64_t max_level;
    OrthMethod method;
    std::string method_tag;
};

static void test_wpt_forward(WPTestCase const& tc) {
    std::string const tag = "wpt_" + tc.dimension + "_" + tc.wavelet + "_" +
        std::to_string(tc.length) + "_L" + std::to_string(tc.max_level);

    auto signal = load_tensor(data_path(tag + "_signal.pt"));
    assert(signal.size(0) == tc.length);

    auto result = wavelet_packet_forward_1d(signal, tc.wavelet, -1, tc.max_level, tc.method);

    for (int64_t level = 1; level <= tc.max_level; ++level) {
        auto ref = load_tensor(data_path(
            tag + "_" + tc.method_tag + "_l" + std::to_string(level) + ".pt"));
        auto ours = result.select(0, level - 1);  // 1-D: [max_level, N] → [N]

        assert(ours.sizes() == ref.sizes());
        auto diff = (ours - ref).abs().max().item<double>();

        if (diff > TOL) {
            std::cerr << "FAIL " << tag << " " << tc.method_tag
                      << " level " << level << " max diff = " << diff << std::endl;
            assert(false);
        }
        std::cout << "    level " << level << " OK (diff=" << diff << ")" << std::endl;
    }
}

static void test_wpt_roundtrip(WPTestCase const& tc) {
    std::string const tag = "wpt_" + tc.dimension + "_" + tc.wavelet + "_" +
        std::to_string(tc.length) + "_L" + std::to_string(tc.max_level);

    auto signal = load_tensor(data_path(tag + "_signal.pt"));
    auto fwd = wavelet_packet_forward_1d(signal, tc.wavelet, -1, tc.max_level, tc.method);
    auto leaf = fwd.select(0, tc.max_level - 1);  // [N]
    auto reconstructed = wavelet_packet_inverse_1d(leaf, tc.wavelet, -1, tc.max_level, tc.method);

    auto diff = (reconstructed - signal).abs().max().item<double>();
    if (diff > TOL) {
        std::cerr << "FAIL roundtrip " << tag << " " << tc.method_tag
                  << " max diff = " << diff << std::endl;
        assert(false);
    }
    std::cout << "    roundtrip OK (diff=" << diff << ")" << std::endl;
}

int main() {
    std::vector<WPTestCase> const cases = {
        {"1D", "haar", 8, 3, OrthMethod::qr, "qr"},
        {"1D", "haar", 16, 4, OrthMethod::qr, "qr"},
        {"1D", "db2", 16, 3, OrthMethod::qr, "qr"},
        {"1D", "db3", 16, 2, OrthMethod::qr, "qr"},
        {"1D", "db3", 32, 3, OrthMethod::qr, "qr"},
        {"1D", "haar", 8, 3, OrthMethod::gramschmidt, "gs"},
        {"1D", "haar", 16, 4, OrthMethod::gramschmidt, "gs"},
        {"1D", "db2", 16, 3, OrthMethod::gramschmidt, "gs"},
        {"1D", "db3", 16, 2, OrthMethod::gramschmidt, "gs"},
        {"1D", "db3", 32, 3, OrthMethod::gramschmidt, "gs"},
    };

    for (auto const& tc : cases) {
        std::string const label = tc.wavelet + " N=" + std::to_string(tc.length) +
            " L=" + std::to_string(tc.max_level) + " " + tc.method_tag;
        std::cout << "  " << label << ":" << std::endl;
        test_wpt_forward(tc);
        test_wpt_roundtrip(tc);
    }

    std::cout << "All transform reference tests passed." << std::endl;
    return 0;
}
