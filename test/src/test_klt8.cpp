// Tests of klt8::trackLK: the AVX2 kernel against the generic code, the generic code against OpenCV's
// calcOpticalFlowPyrLK, and accuracy on known subpixel shifts with and without an exposure change.

#include <basalt/optical_flow/klt8_lk.h>

#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "gtest/gtest.h"

namespace {

// Smooth texture (a few plane waves and Gaussian blobs), sampled at (x - dx, y - dy), times gain plus bias, 8 bits
cv::Mat texture(int w, int h, float dx, float dy, float gain = 1, float bias = 0) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(0, 1);
  struct Blob {
    float x, y, s, a;
  };
  std::vector<Blob> blobs;
  for (int k = 0; k < 400; k++) blobs.push_back({u(rng) * w, u(rng) * h, 2 + 4 * u(rng), (u(rng) - 0.5f) * 120});
  cv::Mat m(h, w, CV_8U);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      const float px = x - dx, py = y - dy;
      float v = 100 + 25 * std::sin(0.31f * px + 0.11f * py) + 15 * std::sin(0.17f * py - 0.23f * px);
      for (const Blob& b : blobs) {
        const float d2 = (px - b.x) * (px - b.x) + (py - b.y) * (py - b.y);
        if (d2 < 16 * b.s * b.s) v += b.a * std::exp(-d2 / (2 * b.s * b.s));
      }
      m.at<uint8_t>(y, x) = cv::saturate_cast<uint8_t>(gain * v + bias);
    }
  return m;
}

std::vector<cv::Mat> pyramid(const cv::Mat& img) {
  std::vector<cv::Mat> p;
  cv::buildOpticalFlowPyramid(img, p, cv::Size(basalt::klt8::kBorder, basalt::klt8::kBorder), 3, true,
                              cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);
  return p;
}

std::vector<cv::Point2f> points(int w, int h, bool borders) {
  std::vector<cv::Point2f> pts;
  for (int y = 20; y < h - 20; y += 23)
    for (int x = 20; x < w - 20; x += 19) pts.emplace_back(x + 0.37f * (y % 3), y + 0.29f * (x % 5));
  if (borders) {
    for (float v : {-12.f, -3.5f, 0.f, 2.2f})
      for (float t : {5.f, 100.3f, 200.7f}) {
        pts.emplace_back(v, t);
        pts.emplace_back(t, v);
        pts.emplace_back(w - 1 - v, t);
        pts.emplace_back(t, h - 1 - v);
      }
  }
  return pts;
}

// Errors |tracked - prev - (dx, dy)| of the tracked points, sorted
std::vector<double> errors(const std::vector<cv::Point2f>& prev, const std::vector<cv::Point2f>& next,
                           const std::vector<uint8_t>& status, float dx, float dy) {
  std::vector<double> e;
  for (size_t k = 0; k < prev.size(); k++)
    if (status[k]) e.push_back(cv::norm(next[k] - prev[k] - cv::Point2f(dx, dy)));
  std::sort(e.begin(), e.end());
  return e;
}

}  // namespace

TEST(Klt8, SimdMatchesGeneric) {
  const int w = 320, h = 240;
  const auto p1 = pyramid(texture(w, h, 0, 0)), p2 = pyramid(texture(w, h, 2.3f, -1.6f, 1.08f, -4));
  const auto pts = points(w, h, true);
  for (int win : {8, 9}) {
    for (bool gain : {false, true}) {
      basalt::klt8::LkParams params;
      params.window = win;
      params.gain_bias = gain;
      std::vector<cv::Point2f> n1 = pts, n2 = pts;
      std::vector<uint8_t> s1, s2;
      std::vector<float> e1, e2;
      basalt::klt8::trackLK(p1, p2, pts, n1, s1, e1, params);
      basalt::klt8::trackLKGeneric(p1, p2, pts, n2, s2, e2, params);
      int tracked = 0;
      for (size_t k = 0; k < pts.size(); k++) {
        ASSERT_EQ(s1[k], s2[k]) << "point " << k << " window " << win << " gain " << gain;
        tracked += s1[k];
        if (!s1[k]) continue;
        EXPECT_EQ(n1[k].x, n2[k].x);
        EXPECT_EQ(n1[k].y, n2[k].y);
        EXPECT_NEAR(e1[k], e2[k], 1e-3f * std::max(1.f, e2[k]));
      }
      EXPECT_GT(tracked, int(pts.size() * 0.8));
    }
  }
}

TEST(Klt8, MatchesOpenCV) {
  const int w = 320, h = 240;
  const auto p1 = pyramid(texture(w, h, 0, 0)), p2 = pyramid(texture(w, h, 3.1f, 1.4f));
  const auto pts = points(w, h, false);
  for (int win : {8, 9}) {
    basalt::klt8::LkParams params;
    params.window = win;
    std::vector<cv::Point2f> n1 = pts, n2 = pts;
    std::vector<uint8_t> s1, s2;
    std::vector<float> e1, e2;
    basalt::klt8::trackLK(p1, p2, pts, n1, s1, e1, params);
    cv::calcOpticalFlowPyrLK(p1, p2, pts, n2, s2, e2, cv::Size(win, win), 3,
                             cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 5, 0.01),
                             cv::OPTFLOW_USE_INITIAL_FLOW, 1e-4);
    int compared = 0, close = 0;
    for (size_t k = 0; k < pts.size(); k++) {
      if (!s1[k] || !s2[k]) continue;
      compared++;
      close += cv::norm(n1[k] - n2[k]) < 0.01;
    }
    EXPECT_GT(compared, int(pts.size() * 0.9));
    EXPECT_GT(close, int(compared * 0.98)) << "window " << win;
  }
}

TEST(Klt8, SubpixelShiftAndExposureChange) {
  const int w = 320, h = 240;
  const float dx = 1.3f, dy = -0.7f;
  const auto p1 = pyramid(texture(w, h, 0, 0));
  const auto p_same = pyramid(texture(w, h, dx, dy));
  const auto p_brighter = pyramid(texture(w, h, dx, dy, 1.1f, -5));
  const auto pts = points(w, h, false);

  auto median_error = [&](const std::vector<cv::Mat>& p2, bool gain) {
    basalt::klt8::LkParams params;
    params.gain_bias = gain;
    std::vector<cv::Point2f> next = pts;
    std::vector<uint8_t> status;
    std::vector<float> err;
    basalt::klt8::trackLK(p1, p2, pts, next, status, err, params);
    const auto e = errors(pts, next, status, dx, dy);
    EXPECT_GT(e.size(), pts.size() * 0.9);
    return e.at(e.size() / 2);
  };

  // 8-bit quantisation and the 8x8 window leave a few hundredths of a pixel
  EXPECT_LT(median_error(p_same, false), 0.04);
  EXPECT_LT(median_error(p_same, true), 0.04);
  EXPECT_LT(median_error(p_brighter, true), 0.04);
  // Without the gain/bias fit the exposure change pulls the result off
  EXPECT_GT(median_error(p_brighter, false), 2 * median_error(p_brighter, true));
}
