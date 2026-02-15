#include "transform.hpp"
#include "wavelet.hpp"
#include "test_util.hpp"

#include <iostream>
#include <string>
#include <vector>

static constexpr double TOL = 1e-7;

struct WPTestCase2D {
    std::string wavelet_name;
    int64_t H, W;
    int64_t max_level;
    OrthMethod method;
    std::string method_tag;
};

static void test_wpt_forward_2d(WPTestCase2D const& tc) {
    std::string const tag = "wpt_2D_" + tc.wavelet_name + "_" +
        std::to_string(tc.H) + "x" + std::to_string(tc.W) +
        "_L" + std::to_string(tc.max_level);

    auto signal = load_tensor(data_path(tag + "_signal.pt"));
    assert(signal.size(0) == tc.H);
    assert(signal.size(1) == tc.W);

    auto const w = make_wavelet(tc.wavelet_name);
    auto result = wavelet_packet_forward_2d(signal, w, {-2, -1}, tc.max_level, tc.method);

    for (int64_t level = 1; level <= tc.max_level; ++level) {
        auto ref = load_tensor(data_path(
            tag + "_" + tc.method_tag + "_l" + std::to_string(level) + ".pt"));
        auto ours = result.select(0, level - 1);  // [max_level, H, W] → [H, W]

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

static void test_wpt_roundtrip_2d(WPTestCase2D const& tc) {
    std::string const tag = "wpt_2D_" + tc.wavelet_name + "_" +
        std::to_string(tc.H) + "x" + std::to_string(tc.W) +
        "_L" + std::to_string(tc.max_level);

    auto signal = load_tensor(data_path(tag + "_signal.pt"));
    auto const w = make_wavelet(tc.wavelet_name);
    auto fwd = wavelet_packet_forward_2d(signal, w, {-2, -1}, tc.max_level, tc.method);
    auto leaf = fwd.select(0, tc.max_level - 1);  // [H, W]
    auto reconstructed = wavelet_packet_inverse_2d(leaf, w, {-2, -1}, tc.max_level, tc.method);

    auto diff = (reconstructed - signal).abs().max().item<double>();
    if (diff > TOL) {
        std::cerr << "FAIL roundtrip " << tag << " " << tc.method_tag
                  << " max diff = " << diff << std::endl;
        assert(false);
    }
    std::cout << "    roundtrip OK (diff=" << diff << ")" << std::endl;
}

int main() {
    std::vector<WPTestCase2D> const cases = {
        {"haar", 8, 8, 3, OrthMethod::qr, "qr"},
        {"haar", 16, 16, 4, OrthMethod::qr, "qr"},
        {"haar", 16, 8, 3, OrthMethod::qr, "qr"},
        {"db2", 16, 16, 3, OrthMethod::qr, "qr"},
        {"db3", 16, 16, 2, OrthMethod::qr, "qr"},
        {"haar", 8, 8, 3, OrthMethod::gramschmidt, "gs"},
        {"haar", 16, 16, 4, OrthMethod::gramschmidt, "gs"},
        {"haar", 16, 8, 3, OrthMethod::gramschmidt, "gs"},
        {"db2", 16, 16, 3, OrthMethod::gramschmidt, "gs"},
        {"db3", 16, 16, 2, OrthMethod::gramschmidt, "gs"},
    };

    for (auto const& tc : cases) {
        std::string const label = tc.wavelet_name + " " + std::to_string(tc.H) + "x" +
            std::to_string(tc.W) + " L=" + std::to_string(tc.max_level) + " " + tc.method_tag;
        std::cout << "  " << label << ":" << std::endl;
        test_wpt_forward_2d(tc);
        test_wpt_roundtrip_2d(tc);
    }

    std::cout << "All 2D transform reference tests passed." << std::endl;
    return 0;
}
