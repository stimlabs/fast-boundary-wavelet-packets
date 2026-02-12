#include "wavelet.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>

static constexpr double TOL = 1e-12;

static void assert_near(double const a, double const b, char const* msg) {
    if (std::abs(a - b) > TOL) {
        std::cerr << msg << ": expected " << b << ", got " << a << std::endl;
        assert(false);
    }
}

static double dot(std::vector<double> const& a, std::vector<double> const& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

static double norm(std::vector<double> const& v) {
    return std::sqrt(dot(v, v));
}

// Haar filter coefficients match expected values.
static void test_haar_coefficients() {
    auto const w = make_wavelet("haar");
    double const s = std::sqrt(2.0) / 2.0;

    assert(w.name == "haar");
    assert(w.dec_len() == 2);
    assert(w.rec_len() == 2);

    assert_near(w.dec_lo[0],  s, "dec_lo[0]");
    assert_near(w.dec_lo[1],  s, "dec_lo[1]");
    assert_near(w.dec_hi[0], -s, "dec_hi[0]");
    assert_near(w.dec_hi[1],  s, "dec_hi[1]");
    assert_near(w.rec_lo[0],  s, "rec_lo[0]");
    assert_near(w.rec_lo[1],  s, "rec_lo[1]");
    assert_near(w.rec_hi[0],  s, "rec_hi[0]");
    assert_near(w.rec_hi[1], -s, "rec_hi[1]");
}

// "db1" is an alias for "haar".
static void test_db1_alias() {
    auto const w = make_wavelet("db1");
    assert(w.name == "haar");
}

// Low-pass and high-pass filters are orthogonal and unit-norm.
static void test_orthogonality() {
    auto const w = make_wavelet("haar");

    assert_near(dot(w.dec_lo, w.dec_hi), 0.0, "dec orthogonality");
    assert_near(dot(w.rec_lo, w.rec_hi), 0.0, "rec orthogonality");
    assert_near(norm(w.dec_lo), 1.0, "dec_lo unit norm");
    assert_near(norm(w.dec_hi), 1.0, "dec_hi unit norm");
    assert_near(norm(w.rec_lo), 1.0, "rec_lo unit norm");
    assert_near(norm(w.rec_hi), 1.0, "rec_hi unit norm");
}

// Unknown wavelet name throws.
static void test_unknown_wavelet() {
    bool threw = false;
    try {
        make_wavelet("not_a_wavelet");
    } catch (std::invalid_argument const&) {
        threw = true;
    }
    assert(threw);
}

int main() {
    test_haar_coefficients();
    test_db1_alias();
    test_orthogonality();
    test_unknown_wavelet();

    std::cout << "All wavelet tests passed." << std::endl;
    return 0;
}
