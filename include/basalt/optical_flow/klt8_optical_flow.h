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

#include <basalt/imu/preintegration.h>
#include <basalt/optical_flow/klt8_lk.h>
#include <basalt/optical_flow/optical_flow.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <vector>

namespace basalt {

/// Frame-to-frame optical flow on 8-bit images (optical_flow_type "klt8").
///
/// Same pipeline as FrameToFrameOpticalFlow (IMU-predicted guesses, forward-backward check, FAST detection in grid
/// cells, cam0 -> cami stereo matching of new points, epipolar filter), but each camera image is kept as ONE 8-bit
/// pyramid (with its Scharr derivatives), built once per frame and shared by tracking, stereo matching and detection,
/// and points are tracked with a translation-only, fixed-point pyramidal Lucas-Kanade (OpenCV's
/// calcOpticalFlowPyrLK) instead of Basalt's float SE(2) patch tracker. Prefers the 8-bit input
/// (ImageData::img8) and falls back to the high byte of the 16-bit one.
///
/// Not supported (ignored): optical_flow_recall_*, optical_flow_pattern.
class Klt8OpticalFlow final : public OpticalFlowBase {
 public:
  using Scalar = float;
  using Vector2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vector4 = Eigen::Matrix<Scalar, 4, 1>;
  using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;
  using SE3 = Sophus::SE3<Scalar>;

  Klt8OpticalFlow(const VioConfig& conf, const Calibration<double>& cal);
  ~Klt8OpticalFlow() override;

  void processingLoop() override;
  bool wantsImage8() const override { return true; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  //! cv::buildOpticalFlowPyramid output with derivatives: {level 0, its Scharr derivatives, level 1, ...}. The images
  //! are views into buffers with a border, which calcOpticalFlowPyrLK samples instead of clamping.
  struct Pyramid {
    std::vector<cv::Mat> levels;
    cv::Mat input8;  //!< 8-bit copy of a 16-bit input image (only used without ImageData::img8)

    Image<const uint8_t> image0() const;
  };

  //! Reused buffers of one trackPoints call
  struct TrackScratch {
    std::vector<KeypointId> ids;
    std::vector<size_t> idx;
    std::vector<cv::Point2f> pts1, pts2, back, offs;
    std::vector<uint8_t> status_fw, status_bw;
    std::vector<float> err_fw, err_bw;
  };

  IntegratedImuMeasurement<double> processImu(int64_t curr_t_ns);
  void processFrame(int64_t curr_t_ns, OpticalFlowInput::Ptr& new_img_vec);
  void buildPyramid(const ImageData& img, Pyramid& pyr) const;

  //! Tracks keypoint_map_1 from pyr_1 (camera cam1) into pyr_2 (camera cam2): forward LK from the guess, backward LK
  //! from the result, keep the point if both converge, it lands in bounds and unmasked, and it comes back to within
  //! sqrt(optical_flow_max_recovered_dist2) of where it started.
  void trackPoints(const Pyramid& pyr_1, const Pyramid& pyr_2, const Keypoints& keypoint_map_1,
                   Keypoints& keypoint_map_2, Keypoints& guesses, const Masks& masks1, const Masks& masks2,
                   const SE3& T_c1_c2, size_t cam1, size_t cam2);

  //! Pyramidal LK from pyr_1 at pts1 into pyr_2, pts2 in: guesses, out: results
  void trackLK(const Pyramid& pyr_1, const Pyramid& pyr_2, const std::vector<cv::Point2f>& pts1,
               std::vector<cv::Point2f>& pts2, std::vector<uint8_t>& status, std::vector<float>& err) const;

  Keypoints addPointsForCamera(size_t cam_id);
  Masks cam0OverlapCellsMasksForCam(size_t cam_id);
  void addPoints();
  void filterPointsForCam(size_t cam_id);
  void filterPoints();

  //! Recounts the points per grid cell, first dropping the newest points of cells over
  //! optical_flow_klt_max_points_cell (if set)
  void updateCellCounts(size_t cam_id);
  bool cellOf(const Vector2& p, int& cx, int& cy) const;
  int& cellCount(size_t cam_id, const Vector2& p);
  void addKeypoint(size_t cam_id, KeypointId kpid, const Keypoint& kp, float response);
  void addKeypoints(size_t cam_id, const Keypoints& kpts);
  void removeKeypoint(size_t cam_id, KeypointId kpid);

  const Calibration<Scalar> calib;
  std::vector<Matrix4> E;  //!< Essential matrix w.r.t. cam 0

  std::vector<Pyramid> pyramids, old_pyramids;  //!< Per camera; swapped every frame so their buffers are reused
  TrackScratch scratch;
  klt8::LkParams lk_params;
  KeypointsData detected;

  Eigen::aligned_vector<Eigen::MatrixXi> cells;  //!< Number of features in each grid cell

  struct OverlapCells {  //!< Grid cell centres of a camera, unprojected, for cam0OverlapCellsMasksForCam
    bool initialized = false;
    std::vector<std::pair<int, int>> centers;  //!< Pixel coordinates of the centres that unproject
    Eigen::aligned_vector<Vector4> ci_xyzw;    //!< Their unprojections
    SE3 T_c0_ci;
  };
  Eigen::aligned_vector<OverlapCells> overlap_cells;

  const Eigen::Vector3d accel_cov;
  const Eigen::Vector3d gyro_cov;
  const int c, w, h, x_start, x_stop, y_start, y_stop;  //!< Grid properties
};

}  // namespace basalt
