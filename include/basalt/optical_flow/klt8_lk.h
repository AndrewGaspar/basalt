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

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <cstdint>
#include <vector>

namespace basalt::klt8 {

//! Parameters of trackLK
struct LkParams {
  int window = 8;          //!< Side of the square window (pixels); 8 has an AVX2 kernel, others run generic code
  int max_level = 3;       //!< Coarsest pyramid level used (0 = full resolution only)
  int max_iterations = 5;  //!< Gauss-Newton iterations per level at most
  float eps = 0.01;        //!< Stop iterating on a level once the update is shorter than this (pixels)
  float min_eig = 1e-4;    //!< Minimum eigenvalue of the window's normalised gradient matrix (OpenCV's scale)
  bool gain_bias = false;  //!< Fit a gain and bias between the windows in every iteration (exposure changes)
};

//! Pyramidal Lucas-Kanade on 8-bit images, translation only, fixed-point bilinear sampling: the algorithm and the
//! number formats of OpenCV's calcOpticalFlowPyrLK (14-bit interpolation weights, intensities with 5 fractional bits,
//! Scharr derivatives, integer window sums), with an optional gain/bias fit, without its per-call allocations and
//! threading, and with an AVX2 kernel for 8x8 windows.
//!
//! prev_pyr / next_pyr: cv::buildOpticalFlowPyramid output with derivatives ({image 0, derivatives 0, image 1, ...})
//! and a border of at least window + 1 pixels (kBorder) around every level. prev_pyr needs the derivatives, next_pyr's
//! are not used. prev: points in prev_pyr level 0; next: in: initial guesses, out: tracked points (both level 0).
//! status: 1 where the point was tracked at level 0. err: mean absolute intensity difference of the windows at the
//! result (after the gain/bias fit if enabled), or 0 where status is 0.
void trackLK(const std::vector<cv::Mat>& prev_pyr, const std::vector<cv::Mat>& next_pyr,
             const std::vector<cv::Point2f>& prev, std::vector<cv::Point2f>& next, std::vector<uint8_t>& status,
             std::vector<float>& err, const LkParams& params);

//! Same, always with the generic (non-SIMD) code; for tests
void trackLKGeneric(const std::vector<cv::Mat>& prev_pyr, const std::vector<cv::Mat>& next_pyr,
                    const std::vector<cv::Point2f>& prev, std::vector<cv::Point2f>& next, std::vector<uint8_t>& status,
                    std::vector<float>& err, const LkParams& params);

//! Border the pyramids need around every level (cv::buildOpticalFlowPyramid's winSize argument)
constexpr int kBorder = 16;

}  // namespace basalt::klt8
