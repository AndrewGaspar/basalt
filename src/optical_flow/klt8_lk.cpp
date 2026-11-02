/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <basalt/optical_flow/klt8_lk.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace basalt::klt8 {

// Number formats (as in OpenCV's LK): bilinear weights with W_BITS fractional bits; intensities sampled with 5
// fractional bits ("x32"); derivatives are Scharr responses (32x the gradient) of the 8-bit image, interpolated.
static constexpr int W_BITS = 14;
static constexpr float FLT_SCALE = 1.f / (1 << 20);
static constexpr int MAX_WIN = 31;
static constexpr int MAX_N = MAX_WIN * MAX_WIN;
static constexpr float GAIN_MIN = 0.8f;  //!< Gain fit clamped to [GAIN_MIN, 1 / GAIN_MIN]

struct Weights {
  int w00, w01, w10, w11;
};

static Weights weights(float a, float b) {
  const int w00 = int(std::lrint((1.f - a) * (1.f - b) * (1 << W_BITS)));
  const int w01 = int(std::lrint(a * (1.f - b) * (1 << W_BITS)));
  const int w10 = int(std::lrint((1.f - a) * b * (1 << W_BITS)));
  return {w00, w01, w10, (1 << W_BITS) - w00 - w01 - w10};
}

static inline int descale(int x, int n) { return (x + (1 << (n - 1))) >> n; }

//! The window of the previous image at a point: intensities and derivatives (row-major), and their sums
struct Template {
  alignas(32) int16_t i[MAX_N];
  alignas(32) int16_t ix[MAX_N];
  alignas(32) int16_t iy[MAX_N];
  int64_t a11, a12, a22;  //!< Sums of ix * ix, ix * iy, iy * iy
  int64_t si, sii;        //!< Sums of i and i * i (gain/bias)
  int64_t sgx, sgy;       //!< Sums of ix, iy (gain/bias)
  int64_t sigx, sigy;     //!< Sums of i * ix, i * iy (gain/bias)
};

//! Sums over the window of the next image (j) at the current estimate
struct Sums {
  int64_t sdx, sdy;  //!< Sums of (j - i) * ix, (j - i) * iy
  int64_t sj, sjj;   //!< Sums of j and j * j (gain/bias)
};

static const uint8_t* px(const cv::Mat& m, int x, int y) { return m.data + ptrdiff_t(y) * ptrdiff_t(m.step[0]) + x; }
static const int16_t* dpx(const cv::Mat& m, int x, int y) {
  return reinterpret_cast<const int16_t*>(m.data + ptrdiff_t(y) * ptrdiff_t(m.step[0])) + 2 * x;
}

// --- Generic code (any window size; defines the arithmetic the SIMD kernels reproduce exactly) ---

static void extractGeneric(const cv::Mat& I, const cv::Mat& dI, int x0, int y0, const Weights& w, int W, Template& t) {
  t.a11 = t.a12 = t.a22 = t.si = t.sii = t.sgx = t.sgy = t.sigx = t.sigy = 0;
  for (int y = 0; y < W; y++) {
    const uint8_t* s0 = px(I, x0, y0 + y);
    const uint8_t* s1 = px(I, x0, y0 + y + 1);
    const int16_t* d0 = dpx(dI, x0, y0 + y);
    const int16_t* d1 = dpx(dI, x0, y0 + y + 1);
    for (int x = 0; x < W; x++) {
      const int k = y * W + x;
      const int ival = descale(s0[x] * w.w00 + s0[x + 1] * w.w01 + s1[x] * w.w10 + s1[x + 1] * w.w11, W_BITS - 5);
      const int ixval =
          descale(d0[2 * x] * w.w00 + d0[2 * x + 2] * w.w01 + d1[2 * x] * w.w10 + d1[2 * x + 2] * w.w11, W_BITS);
      const int iyval = descale(
          d0[2 * x + 1] * w.w00 + d0[2 * x + 3] * w.w01 + d1[2 * x + 1] * w.w10 + d1[2 * x + 3] * w.w11, W_BITS);
      t.i[k] = int16_t(ival);
      t.ix[k] = int16_t(ixval);
      t.iy[k] = int16_t(iyval);
      t.a11 += ixval * ixval;
      t.a12 += ixval * iyval;
      t.a22 += iyval * iyval;
      t.si += ival;
      t.sii += ival * ival;
      t.sgx += ixval;
      t.sgy += iyval;
      t.sigx += int64_t(ival) * ixval;
      t.sigy += int64_t(ival) * iyval;
    }
  }
}

static inline int sampleGeneric(const uint8_t* s0, const uint8_t* s1, int x, const Weights& w) {
  return descale(s0[x] * w.w00 + s0[x + 1] * w.w01 + s1[x] * w.w10 + s1[x + 1] * w.w11, W_BITS - 5);
}

static void sumsGeneric(const cv::Mat& J, int x0, int y0, const Weights& w, int W, const Template& t, Sums& s) {
  s.sdx = s.sdy = s.sj = s.sjj = 0;
  for (int y = 0; y < W; y++) {
    const uint8_t* s0 = px(J, x0, y0 + y);
    const uint8_t* s1 = px(J, x0, y0 + y + 1);
    for (int x = 0; x < W; x++) {
      const int k = y * W + x;
      const int jval = sampleGeneric(s0, s1, x, w);
      const int diff = jval - t.i[k];
      s.sdx += diff * t.ix[k];
      s.sdy += diff * t.iy[k];
      s.sj += jval;
      s.sjj += jval * jval;
    }
  }
}

//! Sum of |j - (a * i + b)| (x32 units), or of |j - i| without gain
static float absResidualGeneric(const cv::Mat& J, int x0, int y0, const Weights& w, int W, const Template& t, bool gain,
                                float a, float b) {
  float sum = 0;
  for (int y = 0; y < W; y++) {
    const uint8_t* s0 = px(J, x0, y0 + y);
    const uint8_t* s1 = px(J, x0, y0 + y + 1);
    for (int x = 0; x < W; x++) {
      const int k = y * W + x;
      const int jval = sampleGeneric(s0, s1, x, w);
      sum += gain ? std::abs(float(jval) - (a * float(t.i[k]) + b)) : float(std::abs(jval - t.i[k]));
    }
  }
  return sum;
}

// --- AVX2 kernels for 8x8 windows ---

#ifdef __AVX2__
static inline __m256i load2rows8(const uint8_t* a, const uint8_t* b) {
  const __m128i lo = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a));
  const __m128i hi = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b));
  return _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(lo, hi));
}

//! Rows r0 and r1 (8 pixels each, starting at the pointers) interpolated with the rows below them (r1, r2):
//! 16 intensities with 5 fractional bits, row r0 in the low half
static inline __m256i interp8x2(const uint8_t* r0, const uint8_t* r1, const uint8_t* r2, __m256i w01, __m256i w23,
                                __m256i delta) {
  const __m256i a = load2rows8(r0, r1), b = load2rows8(r0 + 1, r1 + 1);
  const __m256i c = load2rows8(r1, r2), d = load2rows8(r1 + 1, r2 + 1);
  __m256i lo = _mm256_add_epi32(_mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), w01),
                                _mm256_madd_epi16(_mm256_unpacklo_epi16(c, d), w23));
  __m256i hi = _mm256_add_epi32(_mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), w01),
                                _mm256_madd_epi16(_mm256_unpackhi_epi16(c, d), w23));
  lo = _mm256_srai_epi32(_mm256_add_epi32(lo, delta), W_BITS - 5);
  hi = _mm256_srai_epi32(_mm256_add_epi32(hi, delta), W_BITS - 5);
  return _mm256_packs_epi32(lo, hi);
}

//! Derivatives of one row (8 pixels from d0, interpolated with the row d1 below): low 128 bits = 8 x derivatives,
//! high 128 bits = 8 y derivatives
static inline __m256i interpDeriv8(const int16_t* d0, const int16_t* d1, __m256i w01, __m256i w23, __m256i delta,
                                   __m256i deinterleave) {
  const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d0));
  const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d0 + 2));
  const __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d1));
  const __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d1 + 2));
  // Per 128-bit lane (pixels 0-3 | 4-7): lo = (x, y) of pixels 0, 1 | 4, 5; hi = 2, 3 | 6, 7
  __m256i lo = _mm256_add_epi32(_mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), w01),
                                _mm256_madd_epi16(_mm256_unpacklo_epi16(c, d), w23));
  __m256i hi = _mm256_add_epi32(_mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), w01),
                                _mm256_madd_epi16(_mm256_unpackhi_epi16(c, d), w23));
  lo = _mm256_srai_epi32(_mm256_add_epi32(lo, delta), W_BITS);
  hi = _mm256_srai_epi32(_mm256_add_epi32(hi, delta), W_BITS);
  __m256i v = _mm256_packs_epi32(lo, hi);    // (x, y) of pixels 0-3 | 4-7
  v = _mm256_shuffle_epi8(v, deinterleave);  // x 0-3, y 0-3 | x 4-7, y 4-7
  return _mm256_permute4x64_epi64(v, 0xd8);  // x 0-7 | y 0-7
}

static inline int64_t hsum64(__m256i v) {  // Sum of 8 int32 lanes
  const __m256i a = _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(v)),
                                     _mm256_cvtepi32_epi64(_mm256_extracti128_si256(v, 1)));
  const __m128i b = _mm_add_epi64(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
  return _mm_cvtsi128_si64(b) + _mm_extract_epi64(b, 1);
}

static inline __m256i pairWeights(int lo, int hi) {
  return _mm256_set1_epi32(int((uint32_t(hi) << 16) | uint16_t(lo)));
}

static void extract8(const cv::Mat& I, const cv::Mat& dI, int x0, int y0, const Weights& w, Template& t) {
  const __m256i w01 = pairWeights(w.w00, w.w01), w23 = pairWeights(w.w10, w.w11);
  const __m256i delta_i = _mm256_set1_epi32(1 << (W_BITS - 5 - 1)), delta_d = _mm256_set1_epi32(1 << (W_BITS - 1));
  const __m256i deint = _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,  //
                                         0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
  const __m256i ones = _mm256_set1_epi16(1);
  __m256i a11 = _mm256_setzero_si256(), a12 = a11, a22 = a11, si = a11, sii = a11, sgx = a11, sgy = a11, sigx = a11,
          sigy = a11;
  __m256i dprev = interpDeriv8(dpx(dI, x0, y0), dpx(dI, x0, y0 + 1), w01, w23, delta_d, deint);
  for (int c = 0; c < 4; c++) {
    const int y = y0 + 2 * c;
    const __m256i iv = interp8x2(px(I, x0, y), px(I, x0, y + 1), px(I, x0, y + 2), w01, w23, delta_i);
    const __m256i dnext = interpDeriv8(dpx(dI, x0, y + 1), dpx(dI, x0, y + 2), w01, w23, delta_d, deint);
    const __m256i ix = _mm256_permute2x128_si256(dprev, dnext, 0x20);
    const __m256i iy = _mm256_permute2x128_si256(dprev, dnext, 0x31);
    if (c < 3) dprev = interpDeriv8(dpx(dI, x0, y + 2), dpx(dI, x0, y + 3), w01, w23, delta_d, deint);
    _mm256_store_si256(reinterpret_cast<__m256i*>(t.i + 16 * c), iv);
    _mm256_store_si256(reinterpret_cast<__m256i*>(t.ix + 16 * c), ix);
    _mm256_store_si256(reinterpret_cast<__m256i*>(t.iy + 16 * c), iy);
    a11 = _mm256_add_epi32(a11, _mm256_madd_epi16(ix, ix));
    a12 = _mm256_add_epi32(a12, _mm256_madd_epi16(ix, iy));
    a22 = _mm256_add_epi32(a22, _mm256_madd_epi16(iy, iy));
    si = _mm256_add_epi32(si, _mm256_madd_epi16(iv, ones));
    sii = _mm256_add_epi32(sii, _mm256_madd_epi16(iv, iv));
    sgx = _mm256_add_epi32(sgx, _mm256_madd_epi16(ix, ones));
    sgy = _mm256_add_epi32(sgy, _mm256_madd_epi16(iy, ones));
    sigx = _mm256_add_epi32(sigx, _mm256_madd_epi16(iv, ix));
    sigy = _mm256_add_epi32(sigy, _mm256_madd_epi16(iv, iy));
  }
  t.a11 = hsum64(a11);
  t.a12 = hsum64(a12);
  t.a22 = hsum64(a22);
  t.si = hsum64(si);
  t.sii = hsum64(sii);
  t.sgx = hsum64(sgx);
  t.sgy = hsum64(sgy);
  t.sigx = hsum64(sigx);
  t.sigy = hsum64(sigy);
}

static void sums8(const cv::Mat& J, int x0, int y0, const Weights& w, const Template& t, bool gain, Sums& s) {
  const __m256i w01 = pairWeights(w.w00, w.w01), w23 = pairWeights(w.w10, w.w11);
  const __m256i delta = _mm256_set1_epi32(1 << (W_BITS - 5 - 1));
  const __m256i ones = _mm256_set1_epi16(1);
  __m256i sdx = _mm256_setzero_si256(), sdy = sdx, sj = sdx, sjj = sdx;
  for (int c = 0; c < 4; c++) {
    const int y = y0 + 2 * c;
    const __m256i jv = interp8x2(px(J, x0, y), px(J, x0, y + 1), px(J, x0, y + 2), w01, w23, delta);
    const __m256i iv = _mm256_load_si256(reinterpret_cast<const __m256i*>(t.i + 16 * c));
    const __m256i d = _mm256_sub_epi16(jv, iv);
    sdx =
        _mm256_add_epi32(sdx, _mm256_madd_epi16(d, _mm256_load_si256(reinterpret_cast<const __m256i*>(t.ix + 16 * c))));
    sdy =
        _mm256_add_epi32(sdy, _mm256_madd_epi16(d, _mm256_load_si256(reinterpret_cast<const __m256i*>(t.iy + 16 * c))));
    if (gain) {
      sj = _mm256_add_epi32(sj, _mm256_madd_epi16(jv, ones));
      sjj = _mm256_add_epi32(sjj, _mm256_madd_epi16(jv, jv));
    }
  }
  s.sdx = hsum64(sdx);
  s.sdy = hsum64(sdy);
  s.sj = gain ? hsum64(sj) : 0;
  s.sjj = gain ? hsum64(sjj) : 0;
}

static float absResidual8(const cv::Mat& J, int x0, int y0, const Weights& w, const Template& t, bool gain, float a,
                          float b) {
  const __m256i w01 = pairWeights(w.w00, w.w01), w23 = pairWeights(w.w10, w.w11);
  const __m256i delta = _mm256_set1_epi32(1 << (W_BITS - 5 - 1));
  if (!gain) {
    __m256i acc = _mm256_setzero_si256();
    for (int c = 0; c < 4; c++) {
      const int y = y0 + 2 * c;
      const __m256i jv = interp8x2(px(J, x0, y), px(J, x0, y + 1), px(J, x0, y + 2), w01, w23, delta);
      const __m256i d = _mm256_sub_epi16(jv, _mm256_load_si256(reinterpret_cast<const __m256i*>(t.i + 16 * c)));
      acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_abs_epi16(d), _mm256_set1_epi16(1)));
    }
    return float(hsum64(acc));
  }
  const __m256 va = _mm256_set1_ps(a), vb = _mm256_set1_ps(b);
  const __m256 sign = _mm256_set1_ps(-0.f);
  __m256 acc = _mm256_setzero_ps();
  for (int c = 0; c < 4; c++) {
    const int y = y0 + 2 * c;
    const __m256i jv = interp8x2(px(J, x0, y), px(J, x0, y + 1), px(J, x0, y + 2), w01, w23, delta);
    const __m256i iv = _mm256_load_si256(reinterpret_cast<const __m256i*>(t.i + 16 * c));
    for (int h = 0; h < 2; h++) {
      const __m128i j16 = h ? _mm256_extracti128_si256(jv, 1) : _mm256_castsi256_si128(jv);
      const __m128i i16 = h ? _mm256_extracti128_si256(iv, 1) : _mm256_castsi256_si128(iv);
      const __m256 jf = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(j16));
      const __m256 model = _mm256_add_ps(_mm256_mul_ps(va, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(i16))), vb);
      acc = _mm256_add_ps(acc, _mm256_andnot_ps(sign, _mm256_sub_ps(jf, model)));
    }
  }
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  return _mm_cvtss_f32(s);
}
#endif

// --- The tracker ---

template <bool SIMD8>
static void trackOne(const std::vector<cv::Mat>& P, const std::vector<cv::Mat>& N, int max_level, const LkParams& p,
                     const cv::Point2f& prev, cv::Point2f& next, uint8_t& status, float& err) {
  const int W = p.window;
  const int n = W * W;
  const float half = (W - 1) * 0.5f;
  const float eps2 = p.eps * p.eps;
  Template t;

  status = 1;
  err = 0;
  cv::Point2f next_l;  // Estimate at the current level
  for (int level = max_level; level >= 0; level--) {
    const cv::Mat& I = P[2 * level];
    const cv::Mat& dI = P[2 * level + 1];
    const cv::Mat& J = N[2 * level];
    const float scale = 1.f / float(1 << level);

    next_l = level == max_level ? next * scale : next_l * 2.f;

    const cv::Point2f prev_l = prev * scale - cv::Point2f(half, half);
    const int ix0 = int(std::floor(prev_l.x)), iy0 = int(std::floor(prev_l.y));
    if (ix0 < -W || ix0 >= I.cols || iy0 < -W || iy0 >= I.rows) {
      if (level == 0) status = 0;
      continue;
    }
    const Weights wi = weights(prev_l.x - ix0, prev_l.y - iy0);
#ifdef __AVX2__
    if (SIMD8) extract8(I, dI, ix0, iy0, wi, t);
    else
#endif
      extractGeneric(I, dI, ix0, iy0, wi, W, t);

    // Gain/bias: j ~ a * i + b over the window, a from the ratio of the windows' standard deviations (a regression
    // slope would shrink with the noise of these dark images), b from the means, both refitted every iteration. The
    // residual r = j - a i - b then has zero mean, so the bias direction is projected out of the gradients for the
    // step: A = sum(g' g'^T), g' = g - mean(g) (a window that only ramps cannot be told from a bias change).
    // On synthetic shifts of real frames with a 6-10 % exposure change this keeps the error at the no-change level
    // (median 0.06 px) where plain LK gets 0.3-0.6 px; without a change it costs ~15 % in precision.
    const double si_n = double(t.si) / n;
    const double var_i = double(t.sii) - double(t.si) * si_n;
    const bool gain = p.gain_bias && var_i > 1024.0 * n;  // Not for flat windows (std. dev. < 1 intensity level)
    float A11, A12, A22;
    if (gain) {
      A11 = float(double(t.a11) - double(t.sgx) * double(t.sgx) / n) * FLT_SCALE;
      A12 = float(double(t.a12) - double(t.sgx) * double(t.sgy) / n) * FLT_SCALE;
      A22 = float(double(t.a22) - double(t.sgy) * double(t.sgy) / n) * FLT_SCALE;
    } else {
      A11 = float(t.a11) * FLT_SCALE;
      A12 = float(t.a12) * FLT_SCALE;
      A22 = float(t.a22) * FLT_SCALE;
    }
    float D = A11 * A22 - A12 * A12;
    const float min_eig = (A22 + A11 - std::sqrt((A11 - A22) * (A11 - A22) + 4.f * A12 * A12)) / (2 * n);
    if (min_eig < p.min_eig || D < FLT_EPSILON) {
      if (level == 0) status = 0;
      continue;
    }
    D = 1.f / D;

    float gain_a = 1, gain_b = 0;  // j ~ gain_a * i + gain_b (x32 units)

    cv::Point2f q = next_l - cv::Point2f(half, half);
    cv::Point2f prev_delta(0, 0);
    bool failed = false;
    for (int it = 0; it < p.max_iterations; it++) {
      const int jx0 = int(std::floor(q.x)), jy0 = int(std::floor(q.y));
      if (jx0 < -W || jx0 >= J.cols || jy0 < -W || jy0 >= J.rows) {
        failed = true;
        break;
      }
      const Weights wj = weights(q.x - jx0, q.y - jy0);
      Sums s;
#ifdef __AVX2__
      if (SIMD8) sums8(J, jx0, jy0, wj, t, gain, s);
      else
#endif
        sumsGeneric(J, jx0, jy0, wj, W, t, s);

      float b1, b2;
      if (gain) {
        const double var_j = double(s.sjj) - double(s.sj) * double(s.sj) / n;
        gain_a = var_j > 0 ? float(std::sqrt(var_j / var_i)) : 1.f;
        gain_a = std::clamp(gain_a, GAIN_MIN, 1.f / GAIN_MIN);
        gain_b = float((double(s.sj) - double(gain_a) * double(t.si)) / n);
        // sum(g r) = sum(g (j - i)) + (1 - a) sum(g i) - b sum(g); the residual's derivative is ~ a g, so the step is
        // -(a^2 A)^-1 a sum(g r)
        const double a = gain_a, b = gain_b;
        b1 = float((double(s.sdx) + (1 - a) * double(t.sigx) - b * double(t.sgx)) / a) * FLT_SCALE;
        b2 = float((double(s.sdy) + (1 - a) * double(t.sigy) - b * double(t.sgy)) / a) * FLT_SCALE;
      } else {
        b1 = float(s.sdx) * FLT_SCALE;
        b2 = float(s.sdy) * FLT_SCALE;
      }

      const cv::Point2f delta((A12 * b2 - A22 * b1) * D, (A12 * b1 - A11 * b2) * D);
      q += delta;
      next_l = q + cv::Point2f(half, half);
      if (delta.ddot(delta) <= eps2) break;
      if (it > 0 && std::abs(delta.x + prev_delta.x) < 0.01f && std::abs(delta.y + prev_delta.y) < 0.01f) {
        next_l -= delta * 0.5f;  // Oscillating: take the midpoint
        break;
      }
      prev_delta = delta;
    }
    if (failed && level == 0) status = 0;

    if (level == 0 && status) {
      const cv::Point2f qe = next_l - cv::Point2f(half, half);
      const int jx0 = int(std::floor(qe.x)), jy0 = int(std::floor(qe.y));
      if (jx0 < -W || jx0 >= J.cols || jy0 < -W || jy0 >= J.rows) {
        status = 0;
      } else {
        const Weights wj = weights(qe.x - jx0, qe.y - jy0);
        float sum;
#ifdef __AVX2__
        if (SIMD8) sum = absResidual8(J, jx0, jy0, wj, t, gain, gain_a, gain_b);
        else
#endif
          sum = absResidualGeneric(J, jx0, jy0, wj, W, t, gain, gain_a, gain_b);
        err = sum / float(32 * n);
      }
    }
  }
  next = next_l;
  if (!status) err = 0;
}

template <bool SIMD8>
static void trackAll(const std::vector<cv::Mat>& prev_pyr, const std::vector<cv::Mat>& next_pyr,
                     const std::vector<cv::Point2f>& prev, std::vector<cv::Point2f>& next, std::vector<uint8_t>& status,
                     std::vector<float>& err, const LkParams& p) {
  const int max_level = std::min({p.max_level, int(prev_pyr.size()) / 2 - 1, int(next_pyr.size()) / 2 - 1});
  status.resize(prev.size());
  err.resize(prev.size());
  next.resize(prev.size());
  if (p.window < 2 || p.window > MAX_WIN || max_level < 0) {
    std::fill(status.begin(), status.end(), 0);
    return;
  }
  for (size_t k = 0; k < prev.size(); k++)
    trackOne<SIMD8>(prev_pyr, next_pyr, max_level, p, prev[k], next[k], status[k], err[k]);
}

void trackLK(const std::vector<cv::Mat>& prev_pyr, const std::vector<cv::Mat>& next_pyr,
             const std::vector<cv::Point2f>& prev, std::vector<cv::Point2f>& next, std::vector<uint8_t>& status,
             std::vector<float>& err, const LkParams& params) {
#ifdef __AVX2__
  if (params.window == 8) {
    trackAll<true>(prev_pyr, next_pyr, prev, next, status, err, params);
    return;
  }
#endif
  trackAll<false>(prev_pyr, next_pyr, prev, next, status, err, params);
}

void trackLKGeneric(const std::vector<cv::Mat>& prev_pyr, const std::vector<cv::Mat>& next_pyr,
                    const std::vector<cv::Point2f>& prev, std::vector<cv::Point2f>& next, std::vector<uint8_t>& status,
                    std::vector<float>& err, const LkParams& params) {
  trackAll<false>(prev_pyr, next_pyr, prev, next, status, err, params);
}

}  // namespace basalt::klt8
