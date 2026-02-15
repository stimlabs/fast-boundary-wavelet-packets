#pragma once

#include <string>
#include <vector>

/// A discrete wavelet defined by its four filter banks.
/// The decomposition (analysis) filters split a signal into lowpass and highpass
/// subbands. The reconstruction (synthesis) filters recombine them.
struct Wavelet {
    std::string name;
    std::vector<double> dec_lo;   // decomposition lowpass  (analysis scaling)
    std::vector<double> dec_hi;   // decomposition highpass (analysis wavelet)
    std::vector<double> rec_lo;   // reconstruction lowpass  (synthesis scaling)
    std::vector<double> rec_hi;   // reconstruction highpass (synthesis wavelet)

    int dec_len() const { return static_cast<int>(dec_lo.size()); }
    int rec_len() const { return static_cast<int>(rec_lo.size()); }
};

Wavelet make_wavelet(std::string const& name);
