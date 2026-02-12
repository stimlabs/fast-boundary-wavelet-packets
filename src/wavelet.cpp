#include "wavelet.h"

#include <cmath>
#include <stdexcept>

static Wavelet make_haar() {
    double const s = std::sqrt(2.0) / 2.0;
    return Wavelet{
        .name = "haar",
        .dec_lo = { s,  s},
        .dec_hi = {-s,  s},
        .rec_lo = { s,  s},
        .rec_hi = { s, -s},
    };
}

Wavelet make_wavelet(std::string const& name) {
    if (name == "haar" || name == "db1") {
        return make_haar();
    }
    throw std::invalid_argument("Unknown wavelet: " + name);
}
