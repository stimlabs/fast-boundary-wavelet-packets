#include "matrix_build.hpp"

#include "orthogonalize.hpp"
#include "sparse_math.hpp"

torch::Tensor construct_a(
    Wavelet const& wavelet,
    int64_t length,
    torch::TensorOptions const& opts) {

    auto dec_lo = torch::tensor(wavelet.dec_lo, opts);
    auto dec_hi = torch::tensor(wavelet.dec_hi, opts);

    auto analysis_lo = construct_strided_conv_matrix(dec_lo, length, 2);
    auto analysis_hi = construct_strided_conv_matrix(dec_hi, length, 2);

    return torch::cat({analysis_lo, analysis_hi});
}

torch::Tensor construct_s(
    Wavelet const& wavelet,
    int64_t length,
    torch::TensorOptions const& opts) {

    // Reconstruction filters are time-reversed (flipped) relative to their
    // stored form to produce the correct synthesis convolution matrix.
    auto rec_lo = torch::tensor(wavelet.rec_lo, opts).flip(0);
    auto rec_hi = torch::tensor(wavelet.rec_hi, opts).flip(0);

    auto synthesis_lo = construct_strided_conv_matrix(rec_lo, length, 2);
    auto synthesis_hi = construct_strided_conv_matrix(rec_hi, length, 2);

    return torch::cat({synthesis_lo, synthesis_hi}).t();
}

torch::Tensor construct_boundary_a(
    Wavelet const& wavelet,
    int64_t length,
    torch::TensorOptions const& opts,
    OrthMethod method) {

    auto analysis = construct_a(wavelet, length, opts);
    return orthogonalize(analysis, wavelet.dec_len(), method);
}

torch::Tensor construct_boundary_s(
    Wavelet const& wavelet,
    int64_t length,
    torch::TensorOptions const& opts,
    OrthMethod method) {

    auto synthesis = construct_s(wavelet, length, opts);
    // S is orthogonalized in transposed form because orthogonalize() operates
    // on rows, and the boundary structure of S appears in its columns.
    auto synthesis_transposed = synthesis.t();
    synthesis_transposed = orthogonalize(synthesis_transposed, wavelet.rec_len(), method);
    return synthesis_transposed.t();
}
