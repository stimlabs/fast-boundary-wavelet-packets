#pragma once

#include <string>
#include <vector>

struct Wavelet {
    std::string name;
    std::vector<double> dec_lo;
    std::vector<double> dec_hi;
    std::vector<double> rec_lo;
    std::vector<double> rec_hi;

    int dec_len() const { return static_cast<int>(dec_lo.size()); }
    int rec_len() const { return static_cast<int>(rec_lo.size()); }
};

Wavelet make_wavelet(std::string const& name);
