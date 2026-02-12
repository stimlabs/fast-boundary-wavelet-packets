#include "matrix_build.h"

#include "sparse_math.h"

torch::Tensor construct_a(
    Wavelet const& wavelet,
    int64_t length,
    torch::TensorOptions const& opts) {

    auto dec_lo = torch::tensor(wavelet.dec_lo, opts);
    auto dec_hi = torch::tensor(wavelet.dec_hi, opts);

    auto analysis_lo = construct_strided_conv_matrix(dec_lo, length, 2).to_dense();
    auto analysis_hi = construct_strided_conv_matrix(dec_hi, length, 2).to_dense();

    return torch::cat({analysis_lo, analysis_hi});
}

torch::Tensor construct_s(
    Wavelet const& wavelet,
    int64_t length,
    torch::TensorOptions const& opts) {

    auto rec_lo = torch::tensor(wavelet.rec_lo, opts).flip(0);
    auto rec_hi = torch::tensor(wavelet.rec_hi, opts).flip(0);

    auto synthesis_lo = construct_strided_conv_matrix(rec_lo, length, 2).to_dense();
    auto synthesis_hi = construct_strided_conv_matrix(rec_hi, length, 2).to_dense();

    return torch::cat({synthesis_lo, synthesis_hi}).t();
}
