#include "transform.hpp"
#include "wavelet.hpp"

#include <torch/extension.h>

namespace {

torch::Tensor forward_wrapper(
    torch::Tensor const& input_signal,
    std::string const& wavelet_name,
    int64_t dim,
    int64_t max_level,
    std::string const& orth_method) {
    return wavelet_packet_forward_1d(
        input_signal, make_wavelet(wavelet_name), dim, max_level,
        parse_orth_method(orth_method));
}

torch::Tensor inverse_wrapper(
    torch::Tensor const& leaf_coeffs,
    std::string const& wavelet_name,
    int64_t dim,
    int64_t max_level,
    std::string const& orth_method) {
    return wavelet_packet_inverse_1d(
        leaf_coeffs, make_wavelet(wavelet_name), dim, max_level,
        parse_orth_method(orth_method));
}

torch::Tensor forward_2d_wrapper(
    torch::Tensor const& input_signal,
    std::string const& wavelet_name,
    std::vector<int64_t> dims,
    int64_t max_level,
    std::string const& orth_method) {
    if (dims.size() != 2) {
        throw std::invalid_argument("dims must have exactly 2 elements");
    }
    return wavelet_packet_forward_2d(
        input_signal, make_wavelet(wavelet_name),
        {dims[0], dims[1]}, max_level,
        parse_orth_method(orth_method));
}

torch::Tensor inverse_2d_wrapper(
    torch::Tensor const& leaf_coeffs,
    std::string const& wavelet_name,
    std::vector<int64_t> dims,
    int64_t max_level,
    std::string const& orth_method) {
    if (dims.size() != 2) {
        throw std::invalid_argument("dims must have exactly 2 elements");
    }
    return wavelet_packet_inverse_2d(
        leaf_coeffs, make_wavelet(wavelet_name),
        {dims[0], dims[1]}, max_level,
        parse_orth_method(orth_method));
}

}  // namespace

PYBIND11_MODULE(fbwp, m) {
    m.doc() = "Fast boundary wavelet packets — C++ / LibTorch implementation";

    m.def("wavelet_packet_forward_1d", &forward_wrapper,
          "Forward 1-D wavelet packet transform",
          py::arg("input_signal"),
          py::arg("wavelet_name"),
          py::arg("dim") = -1,
          py::arg("max_level") = -1,
          py::arg("orth_method") = "qr");

    m.def("wavelet_packet_inverse_1d", &inverse_wrapper,
          "Inverse 1-D wavelet packet transform from leaf-level coefficients",
          py::arg("leaf_coeffs"),
          py::arg("wavelet_name"),
          py::arg("dim") = -1,
          py::arg("max_level") = -1,
          py::arg("orth_method") = "qr");

    m.def("wavelet_packet_forward_2d", &forward_2d_wrapper,
          "Forward 2-D wavelet packet transform (separable)",
          py::arg("input_signal"),
          py::arg("wavelet_name"),
          py::arg("dims") = std::vector<int64_t>{-2, -1},
          py::arg("max_level") = -1,
          py::arg("orth_method") = "qr");

    m.def("wavelet_packet_inverse_2d", &inverse_2d_wrapper,
          "Inverse 2-D wavelet packet transform from leaf-level coefficients (separable)",
          py::arg("leaf_coeffs"),
          py::arg("wavelet_name"),
          py::arg("dims") = std::vector<int64_t>{-2, -1},
          py::arg("max_level") = -1,
          py::arg("orth_method") = "qr");

    m.def("compute_max_level", &compute_max_level,
          "Compute the maximum feasible wavelet packet decomposition level",
          py::arg("signal_length"),
          py::arg("dec_len"));
}
