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
#include <basalt/optical_flow/klt8_optical_flow.h>

#include <basalt/utils/keypoints.h>

#include <opencv2/video/tracking.hpp>

#include <cmath>
#include <set>

namespace basalt {

Klt8OpticalFlow::Klt8OpticalFlow(const VioConfig& conf, const Calibration<double>& cal)
    : OpticalFlowBase(conf),
      calib(cal.cast<Scalar>()),
      accel_cov(cal.dicrete_time_accel_noise_std().array().square()),
      gyro_cov(cal.dicrete_time_gyro_noise_std().array().square()),
      c(config.optical_flow_detection_grid_size),
      w(cal.resolution.at(0).x()),
      h(cal.resolution.at(0).y()),
      x_start((w % c) / 2),
      x_stop(x_start + c * (w / c - 1)),
      y_start((h % c) / 2),
      y_stop(y_start + c * (h / c - 1)) {
  const size_t num_cams = calib.intrinsics.size();

  E.resize(num_cams);
  for (size_t i = 0; i < num_cams; i++) {
    Eigen::Matrix4d Ed;
    SE3 T_i_j = calib.T_i_c[0].inverse() * calib.T_i_c[i];
    computeEssential(T_i_j.cast<double>(), Ed);
    E[i] = Ed.cast<Scalar>();
  }

  latest_state = std::make_shared<PoseVelBiasState<double>>();
  predicted_state = std::make_shared<PoseVelBiasState<double>>();
  pyramids.resize(num_cams);
  old_pyramids.resize(num_cams);
  cells.resize(num_cams);
  overlap_cells.resize(num_cams);
  for (size_t i = 0; i < num_cams; i++) cells.at(i).setZero(h / c + 1, w / c + 1);

  lk_params.window = config.optical_flow_klt_window;
  lk_params.max_level = config.optical_flow_levels;
  lk_params.max_iterations = config.optical_flow_max_iterations;
  lk_params.eps = config.optical_flow_klt_converged_eps;
  lk_params.min_eig = config.optical_flow_klt_min_eig;
  lk_params.gain_bias = config.optical_flow_klt_gain_bias;

  if (config.optical_flow_recall_enable) std::cerr << "klt8 optical flow: optical_flow_recall_enable is ignored\n";
}

Klt8OpticalFlow::~Klt8OpticalFlow() = default;

Image<const uint8_t> Klt8OpticalFlow::Pyramid::image0() const {
  const cv::Mat& l0 = levels.at(0);
  return {l0.ptr<uint8_t>(), size_t(l0.cols), size_t(l0.rows), l0.step[0]};
}

void Klt8OpticalFlow::processingLoop() {
  OpticalFlowInput::Ptr img;

  while (true) {
    input_img_queue.pop(img);

    if (img == nullptr) {
      stats.print();
      if (output_queue) output_queue->push(nullptr);
      finished = true;
      break;
    }
    img->addTime("frontend_frames_received");

    while (input_depth_queue.try_pop(depth_guess)) continue;
    if (show_gui) img->depth_guess = depth_guess;

    while (input_lm_bundle_queue.try_pop(latest_lm_bundle)) continue;

    if (!input_state_queue.empty()) {
      while (input_state_queue.try_pop(latest_state)) continue;  // Flush
      first_state_arrived = true;
    } else if (first_state_arrived) {
      latest_state = std::make_shared<PoseVelBiasState<double>>(*predicted_state);
    }

    if (first_state_arrived) {
      auto pim = processImu(img->t_ns);
      pim.predictState(*latest_state, constants::g, *predicted_state);
    }
    img->addTime("frontend_preintegration_computed");

    processFrame(img->t_ns, img);
  }
}

IntegratedImuMeasurement<double> Klt8OpticalFlow::processImu(int64_t curr_t_ns) {
  int64_t prev_t_ns = t_ns;
  Eigen::Vector3d bg = latest_state->bias_gyro;
  Eigen::Vector3d ba = latest_state->bias_accel;
  IntegratedImuMeasurement<double> pim{prev_t_ns, bg, ba};

  if (input_imu_queue.empty()) return pim;

  auto pop_imu = [&](ImuData<double>::Ptr& data) -> bool {
    ImuData<double>::Ptr sample;
    input_imu_queue.pop(sample);  // Blocking pop
    if (sample == nullptr) return false;

    // Calibrate sample
    Eigen::Vector3f a = calib.calib_accel_bias.getCalibrated(sample->accel.cast<Scalar>());
    Eigen::Vector3f g = calib.calib_gyro_bias.getCalibrated(sample->gyro.cast<Scalar>());

    data = std::make_shared<ImuData<double>>();
    data->t_ns = sample->t_ns;
    data->accel = a.cast<double>();
    data->gyro = g.cast<double>();
    return true;
  };

  ImuData<double>::Ptr data = nullptr;
  if (!pop_imu(data)) return pim;

  while (data->t_ns <= prev_t_ns) {
    if (!pop_imu(data)) return pim;
  }

  while (data->t_ns <= curr_t_ns) {
    pim.integrate(*data, accel_cov, gyro_cov);
    if (!pop_imu(data)) return pim;
  }

  // Pretend last IMU sample before "now" happened now
  if (pim.get_start_t_ns() + pim.get_dt_ns() < curr_t_ns) {
    data->t_ns = curr_t_ns;
    pim.integrate(*data, accel_cov, gyro_cov);
  }

  return pim;
}

void Klt8OpticalFlow::buildPyramid(const ImageData& img, Pyramid& pyr) const {
  cv::Mat src;
  if (img.img8) {
    src = cv::Mat(int(img.img8->h), int(img.img8->w), CV_8U, img.img8->ptr, img.img8->pitch);
  } else {
    const ManagedImage<uint16_t>& im = *img.img;
    pyr.input8.create(int(im.h), int(im.w), CV_8U);
    for (size_t y = 0; y < im.h; y++) {
      const uint16_t* __restrict s = im.RowPtr(y);
      uint8_t* __restrict d = pyr.input8.ptr<uint8_t>(int(y));
      for (size_t x = 0; x < im.w; x++) d[x] = s[x] >> 8;
    }
    src = pyr.input8;
  }

  // Copies level 0 into a buffer with a border (reused: the level Mats keep their buffers across frames)
  cv::buildOpticalFlowPyramid(src, pyr.levels, cv::Size(klt8::kBorder, klt8::kBorder), config.optical_flow_levels, true,
                              cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);
}

void Klt8OpticalFlow::processFrame(int64_t curr_t_ns, OpticalFlowInput::Ptr& new_img_vec) {
  for (const auto& v : new_img_vec->img_data) {
    if (!v.img && !v.img8) return;
  }

  const size_t num_cams = calib.intrinsics.size();

  // pyramids now holds the buffers of two frames ago: rebuild them in place
  std::swap(pyramids, old_pyramids);
  for (size_t i = 0; i < num_cams; i++) buildPyramid(new_img_vec->img_data[i], pyramids[i]);
  new_img_vec->addTime("frontend_pyramid_created");

  auto new_transforms = std::make_shared<OpticalFlowResult>();
  new_transforms->keypoints.resize(num_cams);
  new_transforms->keypoint_responses.resize(num_cams);
  new_transforms->tracking_guesses.resize(num_cams);
  new_transforms->matching_guesses.resize(num_cams);
  new_transforms->recall_guesses.resize(num_cams);
  new_transforms->t_ns = curr_t_ns;

  if (t_ns >= 0) {
    SE3 T_i1 = latest_state->T_w_i.cast<Scalar>();
    SE3 T_i2 = predicted_state->T_w_i.cast<Scalar>();
    for (size_t i = 0; i < num_cams; i++) {
      SE3 T_c1 = T_i1 * calib.T_i_c[i];
      SE3 T_c2 = T_i2 * calib.T_i_c[i];
      SE3 T_c1_c2 = T_c1.inverse() * T_c2;
      trackPoints(old_pyramids[i], pyramids[i], transforms->keypoints[i], new_transforms->keypoints[i],
                  new_transforms->tracking_guesses[i], new_img_vec->masks.at(i), new_img_vec->masks.at(i), T_c1_c2, i,
                  i);
    }
  }
  new_img_vec->addTime("frontend_tracking_ended");

  t_ns = curr_t_ns;
  transforms = new_transforms;
  transforms->input_images = new_img_vec;

  for (size_t i = 0; i < num_cams; i++) updateCellCounts(i);
  transforms->input_images->addTime("frontend_recall_ended");

  const KeypointId first_new_id = last_keypoint_id;
  addPoints();
  filterPoints();
  stats.frame(*transforms, first_new_id);
  transforms->input_images->addTime("frontend_filter_ended");

  if (output_queue && frame_counter % config.optical_flow_skip_frames == 0) {
    transforms->input_images->addTime("frontend_keypoints_pushed");
    output_queue->push(transforms);
  }

  frame_counter++;
}

void Klt8OpticalFlow::trackPoints(const Pyramid& pyr_1, const Pyramid& pyr_2, const Keypoints& keypoint_map_1,
                                  Keypoints& keypoint_map_2, Keypoints& guesses, const Masks& masks1,
                                  const Masks& masks2, const SE3& T_c1_c2, size_t cam1, size_t cam2) {
  keypoint_map_2.clear();
  guesses.clear();

  const bool tracking = cam1 == cam2;
  const bool use_depth = tracking || config.optical_flow_matching_guess_type != MatchingGuessType::SAME_PIXEL;
  const Scalar depth = depth_guess;

  TrackScratch& s = scratch;
  s.ids.clear();
  s.idx.clear();
  s.pts1.clear();
  s.pts2.clear();
  s.offs.clear();

  size_t r = 0;
  for (const auto& [kpid, kp] : keypoint_map_1) {
    const Vector2 t1 = kp.translation();
    s.ids.push_back(kpid);
    const size_t k = r++;

    if (masks1.inBounds(t1.x(), t1.y())) continue;

    Vector2 off{0, 0};
    if (use_depth) {
      Vector2 t2_guess;
      Scalar _;
      calib.projectBetweenCams(t1, depth, t2_guess, _, T_c1_c2, cam1, cam2);
      off = t1 - t2_guess;
    }
    const Vector2 t2 = t1 - off;

    if (show_gui) {
      Keypoint g = kp;
      g.translation() = t2;
      guesses[kpid] = g;
    }

    if (!(t2.x() >= 0 && t2.y() >= 0 && t2.x() < w && t2.y() < h)) continue;

    s.idx.push_back(k);
    s.pts1.emplace_back(t1.x(), t1.y());
    s.pts2.emplace_back(t2.x(), t2.y());
    s.offs.emplace_back(off.x(), off.y());
  }
  if (s.pts1.empty()) return;

  trackLK(pyr_1, pyr_2, s.pts1, s.pts2, s.status_fw, s.err_fw);
  s.back.resize(s.pts2.size());
  for (size_t k = 0; k < s.pts2.size(); k++) s.back[k] = s.pts2[k] + s.offs[k];
  trackLK(pyr_2, pyr_1, s.pts2, s.back, s.status_bw, s.err_bw);

  const float max_err = config.optical_flow_klt_max_error;
  const int margin = 2;
  for (size_t k = 0; k < s.pts1.size(); k++) {
    if (!s.status_fw[k] || !s.status_bw[k]) continue;
    if (max_err > 0 && (s.err_fw[k] > max_err || s.err_bw[k] > max_err)) continue;

    const cv::Point2f& q = s.pts2[k];
    if (!(q.x >= margin && q.y >= margin && q.x < w - margin && q.y < h - margin)) continue;
    if (masks2.inBounds(q.x, q.y)) continue;

    const cv::Point2f d = s.pts1[k] - s.back[k];
    if (d.dot(d) >= config.optical_flow_max_recovered_dist2) continue;

    Keypoint kp = Keypoint::Identity();
    kp.translation() = Vector2(q.x, q.y);
    keypoint_map_2.emplace_hint(keypoint_map_2.end(), s.ids[s.idx[k]], kp);
  }
}

void Klt8OpticalFlow::trackLK(const Pyramid& pyr_1, const Pyramid& pyr_2, const std::vector<cv::Point2f>& pts1,
                              std::vector<cv::Point2f>& pts2, std::vector<uint8_t>& status,
                              std::vector<float>& err) const {
  if (config.optical_flow_klt_opencv) {
    const int win = config.optical_flow_klt_window;
    const cv::TermCriteria crit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, config.optical_flow_max_iterations,
                                config.optical_flow_klt_converged_eps);
    cv::calcOpticalFlowPyrLK(pyr_1.levels, pyr_2.levels, pts1, pts2, status, err, cv::Size(win, win),
                             config.optical_flow_levels, crit, cv::OPTFLOW_USE_INITIAL_FLOW,
                             config.optical_flow_klt_min_eig);
    return;
  }
  klt8::trackLK(pyr_1.levels, pyr_2.levels, pts1, pts2, status, err, lk_params);
}

Keypoints Klt8OpticalFlow::addPointsForCamera(size_t cam_id) {
  detectKeypointsWithCells(pyramids.at(cam_id).image0(), detected, cells.at(cam_id),
                           config.optical_flow_detection_grid_size, config.optical_flow_detection_num_points_cell,
                           config.optical_flow_detection_min_threshold, config.optical_flow_detection_max_threshold,
                           config.optical_flow_image_safe_radius, transforms->input_images->masks.at(cam_id));

  Keypoints new_kpts;
  for (size_t i = 0; i < detected.corners.size(); i++) {
    Keypoint kp = Keypoint::Identity();
    kp.translation() = detected.corners[i].cast<Scalar>();

    addKeypoint(cam_id, last_keypoint_id, kp, detected.corner_responses[i]);
    new_kpts.emplace_hint(new_kpts.end(), last_keypoint_id, kp);
    last_keypoint_id++;
  }
  return new_kpts;
}

Masks Klt8OpticalFlow::cam0OverlapCellsMasksForCam(size_t cam_id) {
  // The centres' unprojections only depend on the calibration: compute them once per camera
  OverlapCells& oc = overlap_cells.at(cam_id);
  if (!oc.initialized) {
    for (int y = y_start + c / 2; y <= y_stop + c / 2; y += c) {
      for (int x = x_start + c / 2; x <= x_stop + c / 2; x += c) {
        Vector2 ci_uv(x, y);
        Vector4 ci_xyzw;
        if (!calib.intrinsics[cam_id].unproject(ci_uv, ci_xyzw)) continue;  // Never a valid projection
        oc.centers.emplace_back(x, y);
        oc.ci_xyzw.push_back(ci_xyzw);
      }
    }
    oc.T_c0_ci = (calib.T_i_c[cam_id].inverse() * calib.T_i_c[0]).inverse();
    oc.initialized = true;
  }

  // Same arithmetic as calib.projectBetweenCams(ci_uv, depth_guess, c0_uv, _, cam_id, 0)
  const Scalar depth = depth_guess;
  Masks masks;
  for (size_t k = 0; k < oc.centers.size(); k++) {
    const auto [x, y] = oc.centers[k];
    Vector4 ci_xyzw = oc.ci_xyzw[k] * depth;
    ci_xyzw.w() = 1;
    Vector4 c0_xyzw = oc.T_c0_ci * ci_xyzw;
    Vector2 c0_uv;
    bool projected = calib.intrinsics[0].project(c0_xyzw, c0_uv);
    bool in_bounds = c0_uv.x() >= 0 && c0_uv.x() < w && c0_uv.y() >= 0 && c0_uv.y() < h;
    if (projected && in_bounds) masks.masks.emplace_back(x - c / 2, y - c / 2, c, c);
  }
  return masks;
}

void Klt8OpticalFlow::addPoints() {
  Masks& ms0 = transforms->input_images->masks.at(0);
  Keypoints kpts0 = addPointsForCamera(0);
  transforms->input_images->addTime("frontend_detection_cam0_ended");

  const size_t num_cams = calib.intrinsics.size();
  for (size_t i = 1; i < num_cams; i++) {
    Masks& ms = transforms->input_images->masks.at(i);
    Keypoints& mgs = transforms->matching_guesses.at(i);

    // Match features on areas that overlap with cam0 using optical flow
    Keypoints kpts;
    SE3 T_c0_ci = calib.T_i_c[0].inverse() * calib.T_i_c[i];
    trackPoints(pyramids.at(0), pyramids.at(i), kpts0, kpts, mgs, ms0, ms, T_c0_ci, 0, i);
    addKeypoints(i, kpts);
  }
  transforms->input_images->addTime("frontend_matching_ended");

  for (size_t i = 1; i < num_cams; i++) {
    // Update masks and detect features on area not overlapping with cam0
    if (!config.optical_flow_detection_nonoverlap) continue;
    transforms->input_images->masks.at(i) += cam0OverlapCellsMasksForCam(i);
    addPointsForCamera(i);
  }
  transforms->input_images->addTime("frontend_detection_cami_ended");
}

void Klt8OpticalFlow::filterPointsForCam(size_t cam_id) {
  std::set<KeypointId> kp_to_remove;

  std::vector<KeypointId> kpids;
  Eigen::aligned_vector<Eigen::Vector2f> proj0, proj1;

  for (const auto& [kpid, affine] : transforms->keypoints.at(cam_id)) {
    auto it = transforms->keypoints.at(0).find(kpid);

    if (it != transforms->keypoints.at(0).end()) {
      proj0.emplace_back(it->second.translation());
      proj1.emplace_back(affine.translation());
      kpids.emplace_back(kpid);
    }
  }

  Eigen::aligned_vector<Eigen::Vector4f> p3d0, p3d1;
  std::vector<bool> p3d0_success, p3d1_success;

  calib.intrinsics[0].unproject(proj0, p3d0, p3d0_success);
  calib.intrinsics[cam_id].unproject(proj1, p3d1, p3d1_success);

  for (size_t i = 0; i < p3d0_success.size(); i++) {
    if (p3d0_success[i] && p3d1_success[i]) {
      const double epipolar_error = std::abs(p3d0[i].transpose() * E[cam_id] * p3d1[i]);
      if (epipolar_error > config.optical_flow_epipolar_error) kp_to_remove.emplace(kpids[i]);
    } else {
      kp_to_remove.emplace(kpids[i]);
    }
  }

  for (KeypointId id : kp_to_remove) removeKeypoint(cam_id, id);
}

void Klt8OpticalFlow::filterPoints() {
  for (size_t i = 1; i < calib.intrinsics.size(); i++) filterPointsForCam(i);
}

bool Klt8OpticalFlow::cellOf(const Vector2& p, int& cx, int& cy) const {
  if (p[0] < x_start || p[1] < y_start || p[0] >= x_stop + c || p[1] >= y_stop + c) return false;
  cx = (p[0] - x_start) / c;
  cy = (p[1] - y_start) / c;
  return true;
}

void Klt8OpticalFlow::updateCellCounts(size_t cam_id) {
  Eigen::MatrixXi& cc = cells.at(cam_id);
  cc.setZero(h / c + 1, w / c + 1);
  const int cap = config.optical_flow_klt_max_points_cell;
  Keypoints& kps = transforms->keypoints.at(cam_id);
  // Keypoint ids grow with time, so this visits the oldest tracks first and drops the newest ones of a full cell
  for (auto it = kps.begin(); it != kps.end();) {
    int x = 0, y = 0;
    if (!cellOf(it->second.translation(), x, y)) {
      ++it;
      continue;
    }
    if (cap > 0 && cc(y, x) >= cap) {
      it = kps.erase(it);
      continue;
    }
    cc(y, x)++;
    ++it;
  }
}

// Like FrameToFrameOpticalFlow, points added or removed during a frame count in the cell their coordinates truncate
// to even outside the grid (the edge cells); updateCellCounts only counts points inside it.
int& Klt8OpticalFlow::cellCount(size_t cam_id, const Vector2& p) {
  int x = (p.x() - x_start) / c;
  int y = (p.y() - y_start) / c;
  return cells.at(cam_id)(y, x);
}

void Klt8OpticalFlow::addKeypoint(size_t cam_id, KeypointId kpid, const Keypoint& kp, float response) {
  cellCount(cam_id, kp.translation())++;
  transforms->keypoint_responses.at(cam_id)[kpid] = response;
  transforms->keypoints.at(cam_id)[kpid] = kp;
}

void Klt8OpticalFlow::addKeypoints(size_t cam_id, const Keypoints& kpts) {
  for (const auto& [kpid, kp] : kpts) cellCount(cam_id, kp.translation())++;
  transforms->keypoints.at(cam_id).insert(kpts.begin(), kpts.end());
}

void Klt8OpticalFlow::removeKeypoint(size_t cam_id, KeypointId kpid) {
  Keypoints& kps = transforms->keypoints.at(cam_id);
  auto it = kps.find(kpid);
  if (it == kps.end()) return;
  cellCount(cam_id, it->second.translation())--;
  kps.erase(it);
}

}  // namespace basalt
