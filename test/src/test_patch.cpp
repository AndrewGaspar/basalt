

#include <basalt/optical_flow/patch.h>
#include <sophus/se2.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

#include "gtest/gtest.h"
#include "test_utils.h"

struct SmoothFunction {
  template <typename Scalar>
  Scalar interp(const Eigen::Matrix<Scalar, 2, 1>& p) const {
    return sin(p[0] / 100.0 + p[1] / 20.0);
  }

  template <typename Scalar>
  Eigen::Matrix<Scalar, 3, 1> interpGrad(const Eigen::Matrix<Scalar, 2, 1>& p) const {
    return Eigen::Matrix<Scalar, 3, 1>(sin(p[0] / 100.0 + p[1] / 20.0), cos(p[0] / 100.0 + p[1] / 20.0) / 100.0,
                                       cos(p[0] / 100.0 + p[1] / 20.0) / 20.0);
  }

  template <typename Derived>
  BASALT_HOST_DEVICE inline bool InBounds(const Eigen::MatrixBase<Derived>& p,
                                          const typename Derived::Scalar border) const {
    UNUSED(p);
    UNUSED(border);
    return true;
  }
};

TEST(Patch, ImageInterpolateGrad) {
  Eigen::Vector2i offset(231, 123);

  SmoothFunction img;

  Eigen::Vector2d pd = offset.cast<double>() + Eigen::Vector2d(0.4, 0.34345);

  Eigen::Vector3d val_grad = img.interpGrad<double>(pd);
  Eigen::Matrix<double, 1, 2> J_x = val_grad.tail<2>();

  test_jacobian(
      "d_res_d_x", J_x,
      [&](const Eigen::Vector2d& x) { return Eigen::Matrix<double, 1, 1>(img.interp<double>(pd + x)); },
      Eigen::Vector2d::Zero(), 1);
}

TEST(Patch, PatchSe2Jac) {
  Eigen::Vector2i offset(231, 123);

  SmoothFunction img_view;

  Eigen::Vector2d pd = offset.cast<double>() + Eigen::Vector2d(0.4, 0.34345);

  using PatternT = basalt::Pattern52<double>;
  using PatchT = basalt::OpticalFlowPatch<double, PatternT>;

  double mean, mean2;

  PatchT::VectorP data, data2;
  PatchT::MatrixP3 J_se2;

  basalt::OpticalFlowPatch<double, basalt::Pattern52<double>>::setDataJacSe2(img_view, pd, mean, data, J_se2);

  basalt::OpticalFlowPatch<double, basalt::Pattern52<double>>::setData(img_view, pd, mean2, data2);

  EXPECT_NEAR(mean, mean2, 1e-8);
  EXPECT_TRUE(data.isApprox(data2));

  test_jacobian(
      "d_res_d_se2", J_se2,
      [&](const Eigen::Vector3d& x) {
        Sophus::SE2d transform = Sophus::SE2d::exp(x);

        double mean3;
        PatchT::VectorP data3;

        basalt::OpticalFlowPatch<double, basalt::Pattern52<double>>::setData(img_view, pd, mean3, data3, &transform);

        return data3;
      },
      Eigen::Vector3d::Zero());
}

namespace {

// Textured 16-bit test image like the pyramid levels the tracker sees (8-bit values << 8, smooth structure + noise).
basalt::ManagedImage<uint16_t> makeTestImage(int w, int h, uint32_t seed) {
  basalt::ManagedImage<uint16_t> img(w, h);
  uint32_t state = seed;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      state = state * 1664525u + 1013904223u;
      const double smooth = 128 + 60 * std::sin(x * 0.37 + y * 0.11) * std::cos(y * 0.23 - x * 0.05);
      const double v = smooth + double(state >> 28) - 8;
      img(x, y) = uint16_t(std::clamp(v, 0.0, 255.0)) << 8;
    }
  }
  return img;
}

}  // namespace

// The (AVX2 when available) patch construction against the generic setDataJacSe2 + Eigen solve, at positions all over
// the image: inside (grid fast path for Pattern51 at most positions), at the borders (partially valid patches) and
// outside.
template <typename PatternT>
void testPatchConstruction() {
  using PatchT = basalt::OpticalFlowPatch<float, PatternT>;
  constexpr int N = PatchT::PATTERN_SIZE;

  const basalt::ManagedImage<uint16_t> mimg = makeTestImage(97, 83, 7);
  const basalt::Image<const uint16_t> img(mimg.ptr, mimg.w, mimg.h, mimg.pitch);

  int num_valid = 0;
  for (float y = -2.f; y < 86.f; y += 1.37f) {
    for (float x = -2.f; x < 100.f; x += 0.83f) {
      for (const float frac : {0.f, 0.5f, 0.25f}) {
        const typename PatchT::Vector2 pos(std::floor(x) + frac, y);
        const PatchT p(img, pos);

        float mean = 0;
        typename PatchT::VectorP data;
        typename PatchT::MatrixP3 J;
        PatchT::setDataJacSe2(img, pos, mean, data, J);
        const Eigen::Matrix3f H = J.transpose() * J;
        Eigen::Matrix3f H_inv = Eigen::Matrix3f::Identity();
        H.ldlt().solveInPlace(H_inv);
        const typename PatchT::Matrix3P hjt = H_inv * J.transpose();
        const bool valid = mean > std::numeric_limits<float>::epsilon() && hjt.array().isFinite().all() &&
                           data.array().isFinite().all();

        ASSERT_EQ(p.valid, valid) << "pos " << pos.transpose();
        if (!valid) continue;
        num_valid++;

        EXPECT_NEAR(p.mean, mean, 1e-5f * mean);
        for (int i = 0; i < N; i++) {
          ASSERT_EQ(p.data[i] >= 0, data[i] >= 0);
          EXPECT_NEAR(p.data[i], data[i], 1e-5f);
        }
        for (int i = N; i < PatchT::PADDED_SIZE; i++) EXPECT_EQ(p.data[i], -1.f);
        // H^-1 J^T of a nearly singular H (a few valid points at the border) depends on the rounding of H; and
        // patches with fewer than half of the points valid can never give a valid residual
        if ((data.array() >= 0).count() <= N / 2) continue;
        const Eigen::Vector3d eig = H.cast<double>().selfadjointView<Eigen::Lower>().eigenvalues();
        if (!(eig[0] > 1e-5 * eig[2])) continue;
        const float scale = hjt.cwiseAbs().maxCoeff();
        EXPECT_LE((p.H_se2_inv_J_se2_T.template leftCols<N>() - hjt).cwiseAbs().maxCoeff(), 1e-4f * scale)
            << "pos " << pos.transpose();
        EXPECT_EQ(p.H_se2_inv_J_se2_T.template rightCols<PatchT::PADDED_SIZE - N>().cwiseAbs().maxCoeff(), 0.f);
      }
    }
  }
  EXPECT_GT(num_valid, 1000);
}

// Large images (byte offsets beyond float precision) and tiny ones take the generic path.
TEST(Patch, PatchLargeAndTinyImages) {
  using PatchT = basalt::OpticalFlowPatch<float, basalt::Pattern51<float>>;
  int num_checked = 0;
  for (const auto &[w, h] : {std::pair{4100, 4200}, std::pair{6, 30}}) {
    const basalt::ManagedImage<uint16_t> mimg = makeTestImage(w, h, 5);
    const basalt::Image<const uint16_t> img(mimg.ptr, mimg.w, mimg.h, mimg.pitch);
    for (const PatchT::Vector2 pos : {PatchT::Vector2(w - 9.3f, h - 8.6f), PatchT::Vector2(w / 2 + 0.5f, h / 2)}) {
      const PatchT p(img, pos);
      float mean = 0;
      PatchT::VectorP data;
      PatchT::MatrixP3 J;
      PatchT::setDataJacSe2(img, pos, mean, data, J);
      ASSERT_EQ(p.valid, mean > std::numeric_limits<float>::epsilon());
      if (!p.valid) continue;
      EXPECT_LT((p.data.head<PatchT::PATTERN_SIZE>() - data).cwiseAbs().maxCoeff(), 1e-5f);

      Eigen::AffineCompact2f transform = Eigen::AffineCompact2f::Identity();
      transform.translation() = pos + PatchT::Vector2(0.3f, -0.2f);
      PatchT::Matrix2P transformed_pattern = PatchT::pattern2;
      transformed_pattern.colwise() += transform.translation();
      PatchT::VectorP res;
      Eigen::Vector3f inc;
      const bool valid = p.residual(img, transformed_pattern, res);
      ASSERT_EQ(p.computeIncrement(img, transform, inc), valid);
      if (!valid) continue;
      num_checked++;
      const Eigen::Vector3f inc_ref = -p.H_se2_inv_J_se2_T.leftCols<PatchT::PATTERN_SIZE>() * res;
      EXPECT_LT((inc - inc_ref).cwiseAbs().maxCoeff(), 1e-4f * (1 + inc_ref.cwiseAbs().maxCoeff()));
    }
  }
  EXPECT_GE(num_checked, 2);
}

TEST(Patch, PatchConstruction51) { testPatchConstruction<basalt::Pattern51<float>>(); }
TEST(Patch, PatchConstruction52) { testPatchConstruction<basalt::Pattern52<float>>(); }
TEST(Patch, PatchConstruction24) { testPatchConstruction<basalt::Pattern24<float>>(); }

// computeIncrement (the AVX2 kernel when available) against residual() and the Eigen product, for warps that move
// the pattern partly or fully outside the image.
TEST(Patch, ComputeIncrement) {
  using PatchT = basalt::OpticalFlowPatch<float, basalt::Pattern51<float>>;
  constexpr int N = PatchT::PATTERN_SIZE;

  const basalt::ManagedImage<uint16_t> mimg = makeTestImage(97, 83, 11);
  const basalt::Image<const uint16_t> img(mimg.ptr, mimg.w, mimg.h, mimg.pitch);

  int num_checked = 0;
  for (float y = 5.f; y < 80.f; y += 2.9f) {
    for (float x = 5.f; x < 93.f; x += 3.1f) {
      const PatchT p(img, PatchT::Vector2(x, y));
      if (!p.valid) continue;

      for (const float angle : {0.f, 0.03f, -0.2f, 1.2f}) {
        for (const float shift : {0.f, 0.7f, -2.3f, 6.f}) {
          Eigen::AffineCompact2f transform;
          transform.linear() = Eigen::Rotation2Df(angle).toRotationMatrix();
          transform.translation() = Eigen::Vector2f(x + shift, y - 0.6f * shift);

          typename PatchT::Matrix2P transformed_pattern = transform.linear() * PatchT::pattern2;
          transformed_pattern.colwise() += transform.translation();
          typename PatchT::VectorP res;
          const bool valid = p.residual(img, transformed_pattern, res);
          const Eigen::Vector3f inc_ref = -p.H_se2_inv_J_se2_T.leftCols<N>() * res;

          Eigen::Vector3f inc;
          float error = -1;
          ASSERT_EQ(p.computeIncrement(img, transform, inc, &error), valid);
          EXPECT_NEAR(error, res.norm(), 1e-5f * (1 + res.norm()));
          if (!valid) continue;
          num_checked++;

          // float rounding of sums over the pattern, in a different order
          const Eigen::Vector3f tol = 1e-5f * (p.H_se2_inv_J_se2_T.leftCols<N>().cwiseAbs() *
                                               (res.cwiseAbs().array() + 2.f).matrix());
          for (int k = 0; k < 3; k++) EXPECT_NEAR(inc[k], inc_ref[k], tol[k]);
        }
      }
    }
  }
  EXPECT_GT(num_checked, 500);
}

TEST(Patch, ApplyIncrement) {
  using PatchT = basalt::OpticalFlowPatch<float, basalt::Pattern51<float>>;

  uint32_t state = 3;
  const auto uniform = [&](float a, float b) {
    state = state * 1664525u + 1013904223u;
    return a + (b - a) * float(state >> 8) / float(1 << 24);
  };

  for (int i = 0; i < 10000; i++) {
    Eigen::AffineCompact2f transform;
    transform.linear() = Eigen::Rotation2Df(uniform(-0.5f, 0.5f)).toRotationMatrix();
    transform.translation() = Eigen::Vector2f(uniform(0, 640), uniform(0, 480));
    const float max_angle = i % 4 == 0 ? 3.f : 0.05f;
    const Eigen::Vector3f inc(uniform(-2, 2), uniform(-2, 2), uniform(-max_angle, max_angle));

    // exact result (double), and what float Sophus gives: its (1 - cos(theta)) / theta loses digits for small theta
    Eigen::AffineCompact2d expected = transform.cast<double>();
    expected *= Sophus::SE2d::exp(inc.cast<double>()).matrix();
    Eigen::AffineCompact2f sophus = transform;
    sophus *= Sophus::SE2f::exp(inc).matrix();
    PatchT::applyIncrement(transform, inc);

    // a few ulps
    const double t_err = (transform.translation().cast<double>() - expected.translation()).cwiseAbs().maxCoeff();
    const double t_err_sophus = (sophus.translation().cast<double>() - expected.translation()).cwiseAbs().maxCoeff();
    EXPECT_LT((transform.linear().cast<double>() - expected.linear()).cwiseAbs().maxCoeff(), 3e-7);
    EXPECT_LE(t_err, std::max(3e-7 * expected.translation().cwiseAbs().maxCoeff(), t_err_sophus));
  }
}
