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
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include <Eigen/Dense>
#include <sophus/se2.hpp>

#include <basalt/image/image.h>
#include <basalt/optical_flow/patterns.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define BASALT_PATCH_AVX2 1
#else
#define BASALT_PATCH_AVX2 0
#endif

namespace basalt {

template <typename Scalar, typename Pattern>
struct OpticalFlowPatch {
  static constexpr int PATTERN_SIZE = Pattern::PATTERN_SIZE;

  // The pattern rounded up to whole 8-float SIMD blocks. The padding entries are never valid: data is -1 there and
  // the columns of H_se2_inv_J_se2_T are 0.
  static constexpr int PADDED_SIZE = (PATTERN_SIZE + 7) / 8 * 8;

  // 8-wide AVX2/FMA kernels for float patches on uint16 images (x86-64-v3). Everything else uses the generic Eigen
  // code. The two agree up to floating-point association (sums over the pattern are accumulated 8 lanes at a time).
  static constexpr bool USE_AVX2 = BASALT_PATCH_AVX2 && std::is_same_v<Scalar, float>;

  typedef Eigen::Matrix<int, 2, 1> Vector2i;

  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
  typedef Eigen::Matrix<Scalar, 1, 2> Vector2T;
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 2, 2> Matrix2;
  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
  typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;
  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 1> VectorP;
  typedef Eigen::Matrix<Scalar, PADDED_SIZE, 1> VectorPadded;

  typedef Eigen::Matrix<Scalar, 2, PATTERN_SIZE> Matrix2P;
  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 2> MatrixP2;
  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 3> MatrixP3;
  typedef Eigen::Matrix<Scalar, 3, PATTERN_SIZE> Matrix3P;
  typedef Eigen::Matrix<Scalar, 3, PADDED_SIZE, Eigen::RowMajor> Matrix3Padded;
  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 4> MatrixP4;
  typedef Eigen::Matrix<int, 2, PATTERN_SIZE> Matrix2Pi;

  static const Matrix2P pattern2;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OpticalFlowPatch() = default;

  OpticalFlowPatch(const Image<const uint16_t> &img, const Vector2 &pos) { setFromImage(img, pos); }

  template <typename ImgT>
  static void setData(const ImgT &img, const Vector2 &pos, Scalar &mean, VectorP &data,
                      const Sophus::SE2<Scalar> *se2 = nullptr) {
    int num_valid_points = 0;
    Scalar sum = 0;

    for (int i = 0; i < PATTERN_SIZE; i++) {
      Vector2 p;
      if (se2) {
        p = pos + (*se2) * pattern2.col(i);
      } else {
        p = pos + pattern2.col(i);
      };

      if (img.InBounds(p, 2)) {
        Scalar val = img.template interp<Scalar>(p);
        data[i] = val;
        sum += val;
        num_valid_points++;
      } else {
        data[i] = -1;
      }
    }

    mean = sum / num_valid_points;
    data /= mean;
  }

  template <typename ImgT>
  static void setDataJacSe2(const ImgT &img, const Vector2 &pos, Scalar &mean, VectorP &data, MatrixP3 &J_se2) {
    int num_valid_points = 0;
    Scalar sum = 0;
    Vector3 grad_sum_se2(0, 0, 0);

    Eigen::Matrix<Scalar, 2, 3> Jw_se2;
    Jw_se2.template topLeftCorner<2, 2>().setIdentity();

    for (int i = 0; i < PATTERN_SIZE; i++) {
      Vector2 p = pos + pattern2.col(i);

      // Fill jacobians with respect to SE2 warp
      Jw_se2(0, 2) = -pattern2(1, i);
      Jw_se2(1, 2) = pattern2(0, i);

      if (img.InBounds(p, 2)) {
        Vector3 valGrad = img.template interpGrad<Scalar>(p);
        data[i] = valGrad[0];
        sum += valGrad[0];
        J_se2.row(i) = valGrad.template tail<2>().transpose() * Jw_se2;
        grad_sum_se2 += J_se2.row(i);
        num_valid_points++;
      } else {
        data[i] = -1;
      }
    }

    mean = sum / num_valid_points;

    const Scalar mean_inv = num_valid_points / sum;

    for (int i = 0; i < PATTERN_SIZE; i++) {
      if (data[i] >= 0) {
        J_se2.row(i) -= grad_sum_se2.transpose() * data[i] / sum;
        data[i] *= mean_inv;
      } else {
        J_se2.row(i).setZero();
      }
    }
    J_se2 *= mean_inv;
  }

  void setFromImage(const Image<const uint16_t> &img, const Vector2 &pos) {
#if BASALT_PATCH_AVX2
    if constexpr (USE_AVX2) {
      if (avx2Usable(img)) {
        setFromImageAvx2(img, pos);
        return;
      }
    }
#endif
    setFromImageGeneric(img, pos);
  }

  inline bool residual(const Image<const uint16_t> &img, const Matrix2P &transformed_pattern, VectorP &residual) const {
    Scalar sum = 0;
    int num_valid_points = 0;

    for (int i = 0; i < PATTERN_SIZE; i++) {
      if (img.InBounds(transformed_pattern.col(i), 2)) {
        residual[i] = img.interp<Scalar>(transformed_pattern.col(i));
        sum += residual[i];
        num_valid_points++;
      } else {
        residual[i] = -1;
      }
    }

    // all-black patch cannot be normalized
    if (sum < std::numeric_limits<Scalar>::epsilon()) {
      residual.setZero();
      return false;
    }

    int num_residuals = 0;

    for (int i = 0; i < PATTERN_SIZE; i++) {
      if (residual[i] >= 0 && data[i] >= 0) {
        const Scalar val = residual[i];
        residual[i] = num_valid_points * val / sum - data[i];
        num_residuals++;

      } else {
        residual[i] = 0;
      }
    }

    return num_residuals > PATTERN_SIZE / 2;
  }

  // One Gauss-Newton step of the inverse-compositional tracker: warps the pattern with `transform` (linear() and
  // translation() of an Eigen 2D affine transform) into `img`, and sets inc = -H^-1 J^T r. Returns false (inc
  // undefined) where residual() would. With out_error, also returns ||r|| (computed even when false is returned).
  template <typename TransformT>
  inline bool computeIncrement(const Image<const uint16_t> &img, const TransformT &transform, Vector3 &inc,
                               Scalar *out_error = nullptr) const {
#if BASALT_PATCH_AVX2
    if constexpr (USE_AVX2) {
      if (avx2Usable(img))
        return computeIncrementAvx2(img, transform.linear(), transform.translation(), inc, out_error);
    }
#endif
    VectorP res;

    Matrix2P transformed_pat = transform.linear().matrix() * pattern2;
    transformed_pat.colwise() += transform.translation();

    const bool res_valid = residual(img, transformed_pat, res);

    if (out_error) *out_error = res.norm();

    if (res_valid) inc = -H_se2_inv_J_se2_T.template leftCols<PATTERN_SIZE>() * res;
    return res_valid;
  }

  // transform = transform * SE2::exp(inc) for an Eigen 2D affine transform, without Sophus objects and 3x3 matrix
  // products. The KLT increments are small rotations: below 0.5 rad, sin and cos come from their Taylor series
  // (error < 1e-8, far below float precision; no libm calls, no divisions); otherwise the formulas of
  // Sophus::SE2::exp, including SO2's normalisation of (cos, sin).
  template <typename TransformT>
  static inline void applyIncrement(TransformT &transform, const Vector3 &inc) {
    using std::abs;
    using std::cos;
    using std::sin;
    using std::sqrt;

    const Scalar theta = inc[2];
    Scalar c, s, sin_theta_by_theta, one_minus_cos_theta_by_theta;
    if (abs(theta) < Scalar(0.5)) {
      // Taylor series (truncation error < 1e-8 relative for |theta| < 0.5)
      const Scalar t2 = theta * theta;
      sin_theta_by_theta = Scalar(1) + t2 * (Scalar(-1. / 6) + t2 * (Scalar(1. / 120) + t2 * Scalar(-1. / 5040)));
      one_minus_cos_theta_by_theta =
          theta * (Scalar(0.5) + t2 * (Scalar(-1. / 24) + t2 * (Scalar(1. / 720) + t2 * Scalar(-1. / 40320))));
      s = theta * sin_theta_by_theta;
      c = Scalar(1) - theta * one_minus_cos_theta_by_theta;
    } else {
      c = cos(theta);
      s = sin(theta);
      const Scalar length = Scalar(sqrt(double(c) * double(c) + double(s) * double(s)));
      c /= length;
      s /= length;
      if (abs(theta) < Sophus::Constants<Scalar>::epsilon()) {
        const Scalar theta_sq = theta * theta;
        sin_theta_by_theta = Scalar(1.) - Scalar(1. / 6.) * theta_sq;
        one_minus_cos_theta_by_theta = Scalar(0.5) * theta - Scalar(1. / 24.) * theta * theta_sq;
      } else {
        sin_theta_by_theta = s / theta;
        one_minus_cos_theta_by_theta = (Scalar(1.) - c) / theta;
      }
    }
    const Scalar tx = sin_theta_by_theta * inc[0] - one_minus_cos_theta_by_theta * inc[1];
    const Scalar ty = one_minus_cos_theta_by_theta * inc[0] + sin_theta_by_theta * inc[1];

    const Scalar l00 = transform.linear()(0, 0), l01 = transform.linear()(0, 1);
    const Scalar l10 = transform.linear()(1, 0), l11 = transform.linear()(1, 1);
    const Scalar t0 = transform.translation().x(), t1 = transform.translation().y();
    transform.translation().x() = mulAdd(l00, tx, mulAdd(l01, ty, t0));
    transform.translation().y() = mulAdd(l10, tx, mulAdd(l11, ty, t1));
    transform.linear()(0, 0) = mulAdd(l01, s, l00 * c);
    transform.linear()(0, 1) = mulAdd(l01, c, l00 * -s);
    transform.linear()(1, 0) = mulAdd(l11, s, l10 * c);
    transform.linear()(1, 1) = mulAdd(l11, c, l10 * -s);
  }

  Vector2 pos = Vector2::Zero();
  VectorPadded data = VectorPadded::Constant(-1);  // negative if the point is not valid

  // MatrixP3 J_se2;  // total jacobian with respect to se2 warp
  // Matrix3 H_se2_inv;
  Matrix3Padded H_se2_inv_J_se2_T = Matrix3Padded::Zero();

  Scalar mean = 0;

  bool valid = false;

 private:
  // AVX2 path: H_se2_inv_J_se2_T * data over the valid points, and their number
  Vector3 hjt_data = Vector3::Zero();
  int num_valid_data = 0;

  // a * b + c, fused where the hardware has FMA (std::fma would be a slow library call otherwise)
  static inline Scalar mulAdd(Scalar a, Scalar b, Scalar c) {
#if defined(FP_FAST_FMAF) && defined(FP_FAST_FMA)
    return std::fma(a, b, c);
#else
    return a * b + c;
#endif
  }

  void setFromImageGeneric(const Image<const uint16_t> &img, const Vector2 &pos) {
    this->pos = pos;

    VectorP data_p;
    MatrixP3 J_se2;

    setDataJacSe2(img, pos, mean, data_p, J_se2);

    Matrix3 H_se2 = J_se2.transpose() * J_se2;
    Matrix3 H_se2_inv;
    H_se2_inv.setIdentity();
    H_se2.ldlt().solveInPlace(H_se2_inv);

    data.setConstant(-1);
    data.template head<PATTERN_SIZE>() = data_p;
    H_se2_inv_J_se2_T.setZero();
    H_se2_inv_J_se2_T.template leftCols<PATTERN_SIZE>() = H_se2_inv * J_se2.transpose();

    // NOTE: while it's very unlikely we get a source patch with all black
    // pixels, since points are usually selected at corners, it doesn't cost
    // much to be safe here.

    // all-black patch cannot be normalized; will result in mean of "zero" and
    // H_se2_inv_J_se2_T will contain "NaN" and data will contain "inf"
    valid = mean > std::numeric_limits<Scalar>::epsilon() && H_se2_inv_J_se2_T.array().isFinite().all() &&
            data.array().isFinite().all();

    hjt_data = H_se2_inv_J_se2_T * data;  // 0 columns where data is -1
    num_valid_data = int((data.array() >= 0).count());
  }

  // H^-1 of the symmetric positive semi-definite 3x3 H_se2. A well-conditioned H (the usual case) is inverted from
  // its cofactors in double, (nearly) the correctly rounded inverse. A singular or badly conditioned H goes through
  // Eigen's LDLT like H_se2.ldlt().solveInPlace(identity) in setFromImageGeneric (same pivoting, same pseudo-inverse
  // of D), with the triangular solves written out: Eigen runs them through its general blocked TRSM kernels, which
  // cost more than the whole 3x3 problem.
  static Matrix3 invertSymmetric3(const Matrix3 &H) {
    {
      const double h00 = H(0, 0), h01 = H(0, 1), h02 = H(0, 2), h11 = H(1, 1), h12 = H(1, 2), h22 = H(2, 2);
      const double c00 = h11 * h22 - h12 * h12;
      const double c01 = h02 * h12 - h01 * h22;
      const double c02 = h01 * h12 - h02 * h11;
      const double det = h00 * c00 + h01 * c01 + h02 * c02;
      // det <= h00 h11 h22 for a positive semi-definite H; NaN and inf fail this too
      if (det > 1e-6 * h00 * h11 * h22 && det < std::numeric_limits<double>::infinity()) {
        const double inv_det = 1.0 / det;
        Matrix3 X;
        X(0, 0) = Scalar(c00 * inv_det);
        X(0, 1) = X(1, 0) = Scalar(c01 * inv_det);
        X(0, 2) = X(2, 0) = Scalar(c02 * inv_det);
        X(1, 1) = Scalar((h00 * h22 - h02 * h02) * inv_det);
        X(1, 2) = X(2, 1) = Scalar((h01 * h02 - h00 * h12) * inv_det);
        X(2, 2) = Scalar((h00 * h11 - h01 * h01) * inv_det);
        return X;
      }
    }
    const Eigen::LDLT<Matrix3> ldlt(H);
    const Matrix3 &L = ldlt.matrixLDLT();
    const auto &P = ldlt.transpositionsP();

    // X = P * I
    Matrix3 X = Matrix3::Identity();
    for (int i = 0; i < 3; i++) X.row(i).swap(X.row(P.coeff(i)));

    for (int c = 0; c < 3; c++) {
      // X = L^-1 X (unit lower triangular)
      X(1, c) -= L(1, 0) * X(0, c);
      X(2, c) -= L(2, 0) * X(0, c) + L(2, 1) * X(1, c);

      // X = D^+ X
      for (int i = 0; i < 3; i++) {
        const Scalar d = L(i, i);
        if (std::abs(d) > (std::numeric_limits<Scalar>::min)()) {
          X(i, c) /= d;
        } else {
          X(i, c) = 0;
        }
      }

      // X = L^-T X (unit upper triangular)
      X(1, c) -= L(2, 1) * X(2, c);
      X(0, c) -= L(1, 0) * X(1, c) + L(2, 0) * X(2, c);
    }

    // X = P^T X
    for (int i = 2; i >= 0; i--) X.row(i).swap(X.row(P.coeff(i)));
    return X;
  }

#if BASALT_PATCH_AVX2
  // The AVX2 kernels read a few pixels around the image origin for lanes outside the image, and compute byte offsets
  // in float (exact below 2^24).
  static bool avx2Usable(const Image<const uint16_t> &img) {
    return img.w >= 8 && img.h >= 8 && size_t(img.pitch) * img.h < (size_t(1) << 24);
  }

  // The pattern as structure-of-arrays, padded to PADDED_SIZE (padding at the origin, masked out by lane_mask).
  // grid: the points lie on a unit-spaced grid of at most GRID_MAX x GRID_MAX cells (Pattern51), point i in cell
  // (grid_col[i], grid_row[i]); a block of 8 points takes its lanes from grid rows block_row_a / block_row_b.
  struct alignas(32) PatternSoA {
    static constexpr int GRID_MAX = 8;
    static constexpr int NB = PADDED_SIZE / 8;

    float x[PADDED_SIZE];
    float y[PADDED_SIZE];
    float neg_y[PADDED_SIZE];
    float lane_mask[PADDED_SIZE];  // all bits set for pattern points, 0 for padding

    float grid_col[PADDED_SIZE];
    float grid_row[PADDED_SIZE];
    int32_t block_col_a[PADDED_SIZE];
    int32_t block_col_b[PADDED_SIZE];
    float block_from_b[PADDED_SIZE];  // mask: lane from row b
    int block_row_a[NB];
    int block_row_b[NB];
    int grid_rows;
    bool grid;
  };

  // a float with all bits set, for SIMD lane masks
  static float allBits() {
    const uint32_t bits = 0xffffffffu;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
  }

  static PatternSoA makePatternSoA() {
    PatternSoA s{};
    float min_x = float(Pattern::pattern2(0, 0)), min_y = float(Pattern::pattern2(1, 0)), max_c = 0;
    for (int i = 0; i < PATTERN_SIZE; i++) {
      min_x = std::min(min_x, float(Pattern::pattern2(0, i)));
      min_y = std::min(min_y, float(Pattern::pattern2(1, i)));
    }
    s.grid = true;
    s.grid_rows = 0;
    for (int i = 0; i < PADDED_SIZE; i++) {
      const bool real = i < PATTERN_SIZE;
      s.x[i] = real ? float(Pattern::pattern2(0, i)) : 0.f;
      s.y[i] = real ? float(Pattern::pattern2(1, i)) : 0.f;
      s.neg_y[i] = -s.y[i];
      s.lane_mask[i] = real ? allBits() : 0.f;
      if (!real) continue;
      const float c = s.x[i] - min_x, r = s.y[i] - min_y;
      if (c != std::floor(c) || r != std::floor(r) || c >= PatternSoA::GRID_MAX || r >= PatternSoA::GRID_MAX) {
        s.grid = false;
        continue;
      }
      s.grid_col[i] = c;
      s.grid_row[i] = r;
      s.grid_rows = std::max(s.grid_rows, int(r) + 1);
      max_c = std::max(max_c, c);
    }
    // the grid path reads GRID_MAX + 3 columns around every row, inside the image only if the pattern spans them
    if (max_c != PatternSoA::GRID_MAX - 1) s.grid = false;
    for (int b = 0; s.grid && b < PatternSoA::NB; b++) {
      int ra = -1, rb = -1;
      for (int l = 0; l < 8; l++) {
        const int i = 8 * b + l;
        if (i >= PATTERN_SIZE) continue;
        const int r = int(s.grid_row[i]);
        if (ra < 0 || r == ra) {
          ra = r;
          s.block_col_a[i] = int(s.grid_col[i]);
        } else if (rb < 0 || r == rb) {
          rb = r;
          s.block_col_b[i] = int(s.grid_col[i]);
          s.block_from_b[i] = allBits();
        } else {
          s.grid = false;  // more than two grid rows in one block
        }
      }
      s.block_row_a[b] = std::max(ra, 0);
      s.block_row_b[b] = std::max(rb, 0);
    }
    return s;
  }

  static const PatternSoA &patternSoA() {
    static const PatternSoA soa = makePatternSoA();
    return soa;
  }

  static inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
  }

  static inline int popcount8(__m256 mask) { return __builtin_popcount(unsigned(_mm256_movemask_ps(mask))); }

  // Lanes with lo <= x < hi_x and lo <= y < hi_y (Image::InBounds with border 2 for float coordinates), and the
  // lane is part of the pattern. False for NaN.
  static inline __m256 inBounds(__m256 x, __m256 y, __m256 lo, __m256 hi_x, __m256 hi_y, __m256 lane) {
    const __m256 mx = _mm256_and_ps(_mm256_cmp_ps(x, lo, _CMP_GE_OQ), _mm256_cmp_ps(x, hi_x, _CMP_LT_OQ));
    const __m256 my = _mm256_and_ps(_mm256_cmp_ps(y, lo, _CMP_GE_OQ), _mm256_cmp_ps(y, hi_y, _CMP_LT_OQ));
    return _mm256_and_ps(_mm256_and_ps(mx, my), lane);
  }

  // Two horizontally adjacent uint16 pixels per lane, (x, y) and (x + 1, y), from the byte offset off. The offsets
  // must be readable in every lane (callers point lanes outside mask at a safe pixel and mask their results out).
  static inline __m256i gatherPair(const char *base, __m256i off, __m256 mask) {
#ifdef BASALT_PATCH_HW_GATHER
    return _mm256_mask_i32gather_epi32(_mm256_setzero_si256(), reinterpret_cast<const int *>(base), off,
                                       _mm256_castps_si256(mask), 1);
#else
    (void)mask;
    // unsigned: the offsets are never negative, and 32-bit unsigned indices need no sign extension
    alignas(32) uint32_t o[8];
    _mm256_store_si256(reinterpret_cast<__m256i *>(o), off);
    const auto ld = [&](int i) {
      int32_t v;
      std::memcpy(&v, base + o[i], sizeof(v));
      return v;
    };
    return _mm256_setr_epi32(ld(0), ld(1), ld(2), ld(3), ld(4), ld(5), ld(6), ld(7));
#endif
  }
  static inline __m256 lowPixel(__m256i pair) {
    return _mm256_cvtepi32_ps(_mm256_and_si256(pair, _mm256_set1_epi32(0xffff)));
  }
  static inline __m256 highPixel(__m256i pair) { return _mm256_cvtepi32_ps(_mm256_srli_epi32(pair, 16)); }

  // Bilinear weights in the order of Image::interp: w00 * p(x, y) + w01 * p(x, y + 1) + w10 * p(x + 1, y) + w11 *
  // p(x + 1, y + 1).
  static inline __m256 bilinear(__m256 w00, __m256 w01, __m256 w10, __m256 w11, __m256 p00, __m256 p01, __m256 p10,
                                __m256 p11) {
    __m256 v = _mm256_mul_ps(w00, p00);
    v = _mm256_fmadd_ps(w01, p01, v);
    v = _mm256_fmadd_ps(w10, p10, v);
    return _mm256_fmadd_ps(w11, p11, v);
  }

  // Value and gradient (Image::interpGrad) at every pattern point, for the points whose 4x4 neighbourhood
  // is inside the image (mask); other lanes are garbage. Per lane: reads 6 pixel pairs at their own position.
  static void interpGradPerLane(const Image<const uint16_t> &img, const float *fx_a, const float *fy_a,
                                const float *dx_a, const float *dy_a, const float *mask_a, float *val, float *grad_x,
                                float *grad_y) {
    constexpr int NB = PADDED_SIZE / 8;
    const __m256 one = _mm256_set1_ps(1.f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 pitch_f = _mm256_set1_ps(float(img.pitch));
    const __m256i pitch_i = _mm256_set1_epi32(int(img.pitch));
    const __m256i two_i = _mm256_set1_epi32(2);
    const char *base = reinterpret_cast<const char *>(img.ptr);
    // pixel (1, 1): its 4x4 neighbourhood read below is inside the image (avx2Usable)
    const __m256i safe_off = _mm256_add_epi32(pitch_i, two_i);

    for (int b = 0; b < NB; b++) {
      const __m256 m = _mm256_load_ps(mask_a + 8 * b);
      const __m256 fx = _mm256_load_ps(fx_a + 8 * b);
      const __m256 fy = _mm256_load_ps(fy_a + 8 * b);
      const __m256 dx = _mm256_load_ps(dx_a + 8 * b);
      const __m256 dy = _mm256_load_ps(dy_a + 8 * b);
      const __m256 ddx = _mm256_sub_ps(one, dx);
      const __m256 ddy = _mm256_sub_ps(one, dy);
      const __m256 w00 = _mm256_mul_ps(ddx, ddy);
      const __m256 w01 = _mm256_mul_ps(ddx, dy);
      const __m256 w10 = _mm256_mul_ps(dx, ddy);
      const __m256 w11 = _mm256_mul_ps(dx, dy);

      // byte offset of pixel (ix, iy), exact in float (avx2Usable)
      const __m256i off = _mm256_blendv_epi8(
          safe_off, _mm256_cvtps_epi32(_mm256_fmadd_ps(fy, pitch_f, _mm256_add_ps(fx, fx))), _mm256_castps_si256(m));
      const __m256i off_m = _mm256_sub_epi32(off, two_i);
      const __m256i off_p = _mm256_add_epi32(off, two_i);

      const __m256i r_m1 = gatherPair(base, _mm256_sub_epi32(off, pitch_i), m);    // (ix, iy-1), (ix+1, iy-1)
      const __m256i r0_m = gatherPair(base, off_m, m);                             // (ix-1, iy), (ix, iy)
      const __m256i r0_p = gatherPair(base, off_p, m);                             // (ix+1, iy), (ix+2, iy)
      const __m256i r1_m = gatherPair(base, _mm256_add_epi32(off_m, pitch_i), m);  // (ix-1, iy+1), (ix, iy+1)
      const __m256i r1_p = gatherPair(base, _mm256_add_epi32(off_p, pitch_i), m);  // (ix+1, iy+1), (ix+2, iy+1)
      const __m256i r2 = gatherPair(base, _mm256_add_epi32(off, _mm256_add_epi32(pitch_i, pitch_i)), m);

      const __m256 p0m1 = lowPixel(r_m1), p1m1 = highPixel(r_m1);
      const __m256 pm10 = lowPixel(r0_m), p00 = highPixel(r0_m);
      const __m256 p10 = lowPixel(r0_p), p20 = highPixel(r0_p);
      const __m256 pm11 = lowPixel(r1_m), p01 = highPixel(r1_m);
      const __m256 p11 = lowPixel(r1_p), p21 = highPixel(r1_p);
      const __m256 p02 = lowPixel(r2), p12 = highPixel(r2);

      _mm256_store_ps(val + 8 * b, bilinear(w00, w01, w10, w11, p00, p01, p10, p11));
      const __m256 res_mx = bilinear(w00, w01, w10, w11, pm10, pm11, p00, p01);
      const __m256 res_px = bilinear(w00, w01, w10, w11, p10, p11, p20, p21);
      const __m256 res_my = bilinear(w00, w01, w10, w11, p0m1, p00, p1m1, p10);
      const __m256 res_py = bilinear(w00, w01, w10, w11, p01, p02, p11, p12);
      _mm256_store_ps(grad_x + 8 * b, _mm256_mul_ps(half, _mm256_sub_ps(res_px, res_mx)));
      _mm256_store_ps(grad_y + 8 * b, _mm256_mul_ps(half, _mm256_sub_ps(res_py, res_my)));
    }
  }

  // Same as interpGradPerLane, for a pattern on a unit grid (PatternSoA::grid) placed so that every point has the
  // same integer offset from its grid cell and the same fractional part (dx, dy), and all points are inside the
  // image: then the pixels are contiguous image rows, read with plain vector loads, and the arithmetic per point is
  // the same as in interpGradPerLane. (ix0, iy0) is the integer position of grid cell (0, 0).
  static void interpGradGrid(const Image<const uint16_t> &img, int ix0, int iy0, float dx_s, float dy_s, float *val,
                             float *grad_x, float *grad_y) {
    constexpr int NB = PADDED_SIZE / 8;
    const PatternSoA &pat = patternSoA();

    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 dx = _mm256_set1_ps(dx_s);
    const __m256 dy = _mm256_set1_ps(dy_s);
    const __m256 ddx = _mm256_set1_ps(1.f - dx_s);
    const __m256 ddy = _mm256_set1_ps(1.f - dy_s);
    const __m256 w00 = _mm256_mul_ps(ddx, ddy);
    const __m256 w01 = _mm256_mul_ps(ddx, dy);
    const __m256 w10 = _mm256_mul_ps(dx, ddy);
    const __m256 w11 = _mm256_mul_ps(dx, dy);

    // image rows iy0 - 1 ... iy0 + rows + 1, each at columns ix0 + k + (0..7) for k = -1, 0, 1, 2
    constexpr int MAX_ROWS = PatternSoA::GRID_MAX;
    __m256 px_row[MAX_ROWS + 3][4];
    const int rows = pat.grid_rows;
    for (int r = 0; r < rows + 3; r++) {
      const uint16_t *row = img.RowPtr(iy0 - 1 + r) + ix0 - 1;
      for (int k = 0; k < 4; k++) {
        px_row[r][k] =
            _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(row + k))));
      }
    }

    alignas(32) float g_val[MAX_ROWS * 8];
    alignas(32) float g_gx[MAX_ROWS * 8];
    alignas(32) float g_gy[MAX_ROWS * 8];
    for (int j = 0; j < rows; j++) {
      const __m256 *rm = px_row[j];  // iy - 1
      const __m256 *r0 = px_row[j + 1];
      const __m256 *r1 = px_row[j + 2];
      const __m256 *r2 = px_row[j + 3];
      _mm256_store_ps(g_val + 8 * j, bilinear(w00, w01, w10, w11, r0[1], r1[1], r0[2], r1[2]));
      const __m256 res_mx = bilinear(w00, w01, w10, w11, r0[0], r1[0], r0[1], r1[1]);
      const __m256 res_px = bilinear(w00, w01, w10, w11, r0[2], r1[2], r0[3], r1[3]);
      const __m256 res_my = bilinear(w00, w01, w10, w11, rm[1], r0[1], rm[2], r0[2]);
      const __m256 res_py = bilinear(w00, w01, w10, w11, r1[1], r2[1], r1[2], r2[2]);
      _mm256_store_ps(g_gx + 8 * j, _mm256_mul_ps(half, _mm256_sub_ps(res_px, res_mx)));
      _mm256_store_ps(g_gy + 8 * j, _mm256_mul_ps(half, _mm256_sub_ps(res_py, res_my)));
    }

    // grid order -> pattern order: every block of 8 pattern points comes from at most two grid rows
    for (int b = 0; b < NB; b++) {
      const int ra = 8 * pat.block_row_a[b], rb = 8 * pat.block_row_b[b];
      const __m256i ia = _mm256_load_si256(reinterpret_cast<const __m256i *>(pat.block_col_a + 8 * b));
      const __m256i ib = _mm256_load_si256(reinterpret_cast<const __m256i *>(pat.block_col_b + 8 * b));
      const __m256 sel = _mm256_load_ps(pat.block_from_b + 8 * b);
      const auto pick = [&](const float *g) {
        return _mm256_blendv_ps(_mm256_permutevar8x32_ps(_mm256_load_ps(g + ra), ia),
                                _mm256_permutevar8x32_ps(_mm256_load_ps(g + rb), ib), sel);
      };
      _mm256_store_ps(val + 8 * b, pick(g_val));
      _mm256_store_ps(grad_x + 8 * b, pick(g_gx));
      _mm256_store_ps(grad_y + 8 * b, pick(g_gy));
    }
  }

  void setFromImageAvx2(const Image<const uint16_t> &img, const Vector2 &pos) {
    constexpr int NB = PADDED_SIZE / 8;
    const PatternSoA &pat = patternSoA();

    this->pos = pos;

    const __m256 zero = _mm256_setzero_ps();
    const __m256 lo = _mm256_set1_ps(2.f);
    const __m256 hi_x = _mm256_set1_ps(float(int(img.w)) - 2.f - 1.f);
    const __m256 hi_y = _mm256_set1_ps(float(int(img.h)) - 2.f - 1.f);
    const __m256 pos_x = _mm256_set1_ps(pos.x());
    const __m256 pos_y = _mm256_set1_ps(pos.y());

    alignas(32) float fx_a[PADDED_SIZE], fy_a[PADDED_SIZE], dx_a[PADDED_SIZE], dy_a[PADDED_SIZE];
    alignas(32) float valid_mask[PADDED_SIZE];

    // Pattern positions as in setDataJacSe2: p = pos + pattern2.col(i), ix = int(p.x), dx = p.x - ix, ...; and
    // whether they all sit on the pattern's grid with one common (dx, dy), inside the image.
    __m256 grid_ok = _mm256_set1_ps(pat.grid ? allBits() : 0.f);
    __m256 dx_ref = zero, dy_ref = zero, gx_ref = zero, gy_ref = zero;
    for (int b = 0; b < NB; b++) {
      const __m256 lane = _mm256_load_ps(pat.lane_mask + 8 * b);
      const __m256 x = _mm256_add_ps(pos_x, _mm256_load_ps(pat.x + 8 * b));
      const __m256 y = _mm256_add_ps(pos_y, _mm256_load_ps(pat.y + 8 * b));
      const __m256 m = inBounds(x, y, lo, hi_x, hi_y, lane);
      const __m256 fx = _mm256_cvtepi32_ps(_mm256_cvttps_epi32(x));
      const __m256 fy = _mm256_cvtepi32_ps(_mm256_cvttps_epi32(y));
      const __m256 dx = _mm256_sub_ps(x, fx);
      const __m256 dy = _mm256_sub_ps(y, fy);
      // integer position of grid cell (0, 0) as seen from each point (exact: small integers in float)
      const __m256 gx = _mm256_sub_ps(fx, _mm256_load_ps(pat.grid_col + 8 * b));
      const __m256 gy = _mm256_sub_ps(fy, _mm256_load_ps(pat.grid_row + 8 * b));
      if (b == 0) {
        dx_ref = _mm256_broadcastss_ps(_mm256_castps256_ps128(dx));
        dy_ref = _mm256_broadcastss_ps(_mm256_castps256_ps128(dy));
        gx_ref = _mm256_broadcastss_ps(_mm256_castps256_ps128(gx));
        gy_ref = _mm256_broadcastss_ps(_mm256_castps256_ps128(gy));
      }
      const __m256 same =
          _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(dx, dx_ref, _CMP_EQ_OQ), _mm256_cmp_ps(dy, dy_ref, _CMP_EQ_OQ)),
                        _mm256_and_ps(_mm256_cmp_ps(gx, gx_ref, _CMP_EQ_OQ), _mm256_cmp_ps(gy, gy_ref, _CMP_EQ_OQ)));
      // padding lanes: always fine
      grid_ok =
          _mm256_and_ps(grid_ok, _mm256_or_ps(_mm256_and_ps(same, m), _mm256_xor_ps(lane, _mm256_set1_ps(allBits()))));

      _mm256_store_ps(fx_a + 8 * b, fx);
      _mm256_store_ps(fy_a + 8 * b, fy);
      _mm256_store_ps(dx_a + 8 * b, dx);
      _mm256_store_ps(dy_a + 8 * b, dy);
      _mm256_store_ps(valid_mask + 8 * b, m);
    }

    alignas(32) float val[PADDED_SIZE];
    alignas(32) float jac[3][PADDED_SIZE];

    if (_mm256_movemask_ps(grid_ok) == 0xff) {
      interpGradGrid(img, int(_mm256_cvtss_f32(gx_ref)), int(_mm256_cvtss_f32(gy_ref)), _mm256_cvtss_f32(dx_ref),
                     _mm256_cvtss_f32(dy_ref), val, jac[0], jac[1]);
    } else {
      interpGradPerLane(img, fx_a, fy_a, dx_a, dy_a, valid_mask, val, jac[0], jac[1]);
    }

    __m256 sum_v = zero, gs0_v = zero, gs1_v = zero, gs2_v = zero;
    int num_valid_points = 0;

    for (int b = 0; b < NB; b++) {
      const __m256 m = _mm256_load_ps(valid_mask + 8 * b);
      const __m256 gx = _mm256_load_ps(jac[0] + 8 * b);
      const __m256 gy = _mm256_load_ps(jac[1] + 8 * b);

      // J_se2 row = [gx gy] * [1 0 -py; 0 1 px]
      const __m256 j0 = _mm256_and_ps(gx, m);
      const __m256 j1 = _mm256_and_ps(gy, m);
      const __m256 j2 = _mm256_and_ps(
          _mm256_fmadd_ps(gy, _mm256_load_ps(pat.x + 8 * b), _mm256_mul_ps(gx, _mm256_load_ps(pat.neg_y + 8 * b))), m);
      const __m256 vm = _mm256_and_ps(_mm256_load_ps(val + 8 * b), m);

      _mm256_store_ps(val + 8 * b, vm);
      _mm256_store_ps(jac[0] + 8 * b, j0);
      _mm256_store_ps(jac[1] + 8 * b, j1);
      _mm256_store_ps(jac[2] + 8 * b, j2);

      sum_v = _mm256_add_ps(sum_v, vm);
      gs0_v = _mm256_add_ps(gs0_v, j0);
      gs1_v = _mm256_add_ps(gs1_v, j1);
      gs2_v = _mm256_add_ps(gs2_v, j2);
      num_valid_points += popcount8(m);
    }

    const float sum = hsum(sum_v);
    mean = sum / num_valid_points;
    const float mean_inv = num_valid_points / sum;

    const __m256 mean_inv_b = _mm256_set1_ps(mean_inv);
    // grad_sum / sum
    const __m256 gs0 = _mm256_set1_ps(hsum(gs0_v) / sum);
    const __m256 gs1 = _mm256_set1_ps(hsum(gs1_v) / sum);
    const __m256 gs2 = _mm256_set1_ps(hsum(gs2_v) / sum);
    const __m256 minus_one = _mm256_set1_ps(-1.f);

    __m256 h00 = zero, h01 = zero, h02 = zero, h11 = zero, h12 = zero, h22 = zero;

    for (int b = 0; b < NB; b++) {
      const __m256 m = _mm256_load_ps(valid_mask + 8 * b);
      const __m256 lane = _mm256_load_ps(pat.lane_mask + 8 * b);
      const __m256 d = _mm256_load_ps(val + 8 * b);

      // valid rows: J -= data * grad_sum^T / sum; invalid rows: 0; then all rows *= mean_inv
      const __m256 j0 = _mm256_and_ps(
          _mm256_mul_ps(_mm256_and_ps(_mm256_fnmadd_ps(d, gs0, _mm256_load_ps(jac[0] + 8 * b)), m), mean_inv_b), lane);
      const __m256 j1 = _mm256_and_ps(
          _mm256_mul_ps(_mm256_and_ps(_mm256_fnmadd_ps(d, gs1, _mm256_load_ps(jac[1] + 8 * b)), m), mean_inv_b), lane);
      const __m256 j2 = _mm256_and_ps(
          _mm256_mul_ps(_mm256_and_ps(_mm256_fnmadd_ps(d, gs2, _mm256_load_ps(jac[2] + 8 * b)), m), mean_inv_b), lane);

      _mm256_store_ps(jac[0] + 8 * b, j0);
      _mm256_store_ps(jac[1] + 8 * b, j1);
      _mm256_store_ps(jac[2] + 8 * b, j2);
      _mm256_storeu_ps(data.data() + 8 * b, _mm256_blendv_ps(minus_one, _mm256_mul_ps(d, mean_inv_b), m));

      h00 = _mm256_fmadd_ps(j0, j0, h00);
      h01 = _mm256_fmadd_ps(j0, j1, h01);
      h02 = _mm256_fmadd_ps(j0, j2, h02);
      h11 = _mm256_fmadd_ps(j1, j1, h11);
      h12 = _mm256_fmadd_ps(j1, j2, h12);
      h22 = _mm256_fmadd_ps(j2, j2, h22);
    }

    Matrix3 H_se2;
    H_se2(0, 0) = hsum(h00);
    H_se2(1, 1) = hsum(h11);
    H_se2(2, 2) = hsum(h22);
    H_se2(0, 1) = H_se2(1, 0) = hsum(h01);
    H_se2(0, 2) = H_se2(2, 0) = hsum(h02);
    H_se2(1, 2) = H_se2(2, 1) = hsum(h12);

    const Matrix3 H_se2_inv = invertSymmetric3(H_se2);

    __m256 not_finite = zero;
    for (int r = 0; r < 3; r++) {
      const __m256 a0 = _mm256_set1_ps(H_se2_inv(r, 0));
      const __m256 a1 = _mm256_set1_ps(H_se2_inv(r, 1));
      const __m256 a2 = _mm256_set1_ps(H_se2_inv(r, 2));
      float *out = H_se2_inv_J_se2_T.row(r).data();
      for (int b = 0; b < NB; b++) {
        const __m256 lane = _mm256_load_ps(pat.lane_mask + 8 * b);
        __m256 h = _mm256_mul_ps(a0, _mm256_load_ps(jac[0] + 8 * b));
        h = _mm256_fmadd_ps(a1, _mm256_load_ps(jac[1] + 8 * b), h);
        h = _mm256_fmadd_ps(a2, _mm256_load_ps(jac[2] + 8 * b), h);
        // x - x is NaN for inf and NaN (padding lanes are 0 here, see the lane mask on J)
        not_finite = _mm256_or_ps(not_finite, _mm256_cmp_ps(_mm256_sub_ps(h, h), zero, _CMP_NEQ_UQ));
        _mm256_storeu_ps(out + 8 * b, _mm256_and_ps(h, lane));
      }
    }
    // H^-1 J^T d for computeIncrementAvx2, accumulated in the same order as there
    __m256 ad0 = zero, ad1 = zero, ad2 = zero;
    for (int b = 0; b < NB; b++) {
      const __m256 d = _mm256_loadu_ps(data.data() + 8 * b);
      not_finite = _mm256_or_ps(not_finite, _mm256_cmp_ps(_mm256_sub_ps(d, d), zero, _CMP_NEQ_UQ));
      const __m256 dm = _mm256_and_ps(d, _mm256_load_ps(pat.lane_mask + 8 * b));
      ad0 = _mm256_fmadd_ps(_mm256_loadu_ps(H_se2_inv_J_se2_T.row(0).data() + 8 * b), dm, ad0);
      ad1 = _mm256_fmadd_ps(_mm256_loadu_ps(H_se2_inv_J_se2_T.row(1).data() + 8 * b), dm, ad1);
      ad2 = _mm256_fmadd_ps(_mm256_loadu_ps(H_se2_inv_J_se2_T.row(2).data() + 8 * b), dm, ad2);
    }
    hjt_data = Vector3(hsum(ad0), hsum(ad1), hsum(ad2));
    num_valid_data = num_valid_points;

    // all-black patch cannot be normalized; see setFromImageGeneric
    valid = mean > std::numeric_limits<Scalar>::epsilon() && _mm256_movemask_ps(not_finite) == 0;
  }

  bool computeIncrementAvx2(const Image<const uint16_t> &img, const Matrix2 &R, const Vector2 &t, Vector3 &inc,
                            Scalar *out_error) const {
    constexpr int NB = PADDED_SIZE / 8;
    const PatternSoA &pat = patternSoA();

    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.f);
    const __m256 lo = _mm256_set1_ps(2.f);
    const __m256 hi_x = _mm256_set1_ps(float(int(img.w)) - 2.f - 1.f);
    const __m256 hi_y = _mm256_set1_ps(float(int(img.h)) - 2.f - 1.f);
    const __m256 pitch_f = _mm256_set1_ps(float(img.pitch));
    const __m256 r00 = _mm256_set1_ps(R(0, 0)), r01 = _mm256_set1_ps(R(0, 1));
    const __m256 r10 = _mm256_set1_ps(R(1, 0)), r11 = _mm256_set1_ps(R(1, 1));
    const __m256 tx = _mm256_set1_ps(t.x()), ty = _mm256_set1_ps(t.y());
    const char *row0 = reinterpret_cast<const char *>(img.ptr);
    const char *row1 = row0 + img.pitch;

    const float *hjt0 = H_se2_inv_J_se2_T.row(0).data();
    const float *hjt1 = H_se2_inv_J_se2_T.row(1).data();
    const float *hjt2 = H_se2_inv_J_se2_T.row(2).data();

    __m256 val[NB];
    __m256 inb[NB];
    __m256 sum_v = zero;
    int num_valid_points = 0;
    __m256 av0 = zero, av1 = zero, av2 = zero;

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC unroll 8
#endif
    for (int b = 0; b < NB; b++) {
      const __m256 px = _mm256_load_ps(pat.x + 8 * b);
      const __m256 py = _mm256_load_ps(pat.y + 8 * b);
      // transform.linear() * pattern2 + transform.translation()
      const __m256 x = _mm256_add_ps(_mm256_fmadd_ps(r01, py, _mm256_mul_ps(r00, px)), tx);
      const __m256 y = _mm256_add_ps(_mm256_fmadd_ps(r11, py, _mm256_mul_ps(r10, px)), ty);
      const __m256 m = inBounds(x, y, lo, hi_x, hi_y, _mm256_load_ps(pat.lane_mask + 8 * b));

      const __m256 fx = _mm256_cvtepi32_ps(_mm256_cvttps_epi32(x));
      const __m256 fy = _mm256_cvtepi32_ps(_mm256_cvttps_epi32(y));
      const __m256 dx = _mm256_sub_ps(x, fx);
      const __m256 dy = _mm256_sub_ps(y, fy);
      const __m256 ddx = _mm256_sub_ps(one, dx);
      const __m256 ddy = _mm256_sub_ps(one, dy);

      // lanes outside the image read pixel (0, 0)
      const __m256i off = _mm256_and_si256(_mm256_cvtps_epi32(_mm256_fmadd_ps(fy, pitch_f, _mm256_add_ps(fx, fx))),
                                           _mm256_castps_si256(m));
      const __m256i g0 = gatherPair(row0, off, m);  // (ix, iy), (ix+1, iy)
      const __m256i g1 = gatherPair(row1, off, m);  // (ix, iy+1), (ix+1, iy+1)

      // Image::interp
      const __m256 v = bilinear(_mm256_mul_ps(ddx, ddy), _mm256_mul_ps(ddx, dy), _mm256_mul_ps(dx, ddy),
                                _mm256_mul_ps(dx, dy), lowPixel(g0), lowPixel(g1), highPixel(g0), highPixel(g1));

      // r = s * v - d with s = n / sum (unknown until all values are in), so accumulate H^-1 J^T v here and
      // H^-1 J^T d below: inc = -(s * H^-1 J^T v - H^-1 J^T d), over the valid residuals (H^-1 J^T is 0 where d < 0).
      const __m256 vm = _mm256_and_ps(v, m);
      av0 = _mm256_fmadd_ps(_mm256_loadu_ps(hjt0 + 8 * b), vm, av0);
      av1 = _mm256_fmadd_ps(_mm256_loadu_ps(hjt1 + 8 * b), vm, av1);
      av2 = _mm256_fmadd_ps(_mm256_loadu_ps(hjt2 + 8 * b), vm, av2);
      val[b] = v;
      inb[b] = m;
      sum_v = _mm256_add_ps(sum_v, vm);
      num_valid_points += popcount8(m);
    }

    const float sum = hsum(sum_v);

    // all-black patch cannot be normalized
    if (sum < std::numeric_limits<Scalar>::epsilon()) {
      if (out_error) *out_error = 0;
      return false;
    }

    const __m256 n_b = _mm256_set1_ps(float(num_valid_points));
    const __m256 sum_b = _mm256_set1_ps(sum);

    if (out_error) {
      __m256 err = zero;
      for (int b = 0; b < NB; b++) {
        const __m256 d = _mm256_loadu_ps(data.data() + 8 * b);
        const __m256 rm = _mm256_and_ps(inb[b], _mm256_cmp_ps(d, zero, _CMP_GE_OQ));
        const __m256 r = _mm256_and_ps(_mm256_sub_ps(_mm256_div_ps(_mm256_mul_ps(n_b, val[b]), sum_b), d), rm);
        err = _mm256_fmadd_ps(r, r, err);
      }
      *out_error = std::sqrt(hsum(err));
    }

    // H^-1 J^T d and the number of residuals: precomputed for the usual case of the whole pattern inside the image
    Vector3 hjt_d = hjt_data;
    int num_residuals = num_valid_data;
    if (num_valid_points != PATTERN_SIZE) {
      __m256 ad0 = zero, ad1 = zero, ad2 = zero;
      num_residuals = 0;
      for (int b = 0; b < NB; b++) {
        const __m256 d = _mm256_loadu_ps(data.data() + 8 * b);
        const __m256 dm = _mm256_and_ps(d, inb[b]);
        ad0 = _mm256_fmadd_ps(_mm256_loadu_ps(hjt0 + 8 * b), dm, ad0);
        ad1 = _mm256_fmadd_ps(_mm256_loadu_ps(hjt1 + 8 * b), dm, ad1);
        ad2 = _mm256_fmadd_ps(_mm256_loadu_ps(hjt2 + 8 * b), dm, ad2);
        num_residuals += popcount8(_mm256_and_ps(inb[b], _mm256_cmp_ps(d, zero, _CMP_GE_OQ)));
      }
      hjt_d = Vector3(hsum(ad0), hsum(ad1), hsum(ad2));
    }

    const float scale = float(num_valid_points) / sum;
    inc = Vector3(hjt_d[0] - scale * hsum(av0), hjt_d[1] - scale * hsum(av1), hjt_d[2] - scale * hsum(av2));

    return num_residuals > PATTERN_SIZE / 2;
  }
#endif
};

template <typename Scalar, typename Pattern>
const typename OpticalFlowPatch<Scalar, Pattern>::Matrix2P OpticalFlowPatch<Scalar, Pattern>::pattern2 =
    Pattern::pattern2;

}  // namespace basalt
