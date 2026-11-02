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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <basalt/image/image_pyr.h>

namespace basalt {

/// Builds a uint16 pyramid with exactly the result of ManagedImagePyr<uint16_t>::setFromImage (same levels,
/// bit-identical pixels), but cheaper:
/// - pyr's buffer is reused when it already has the size (ManagedImagePyr reallocates and zero-fills it
///   every time). Only a fresh buffer is zero-filled: the levels never write the rest of the mipmap.
/// - the subsampling walks rows (one int row of vertical sums per output row, then the horizontal pass on
///   it), where ManagedImagePyr::subsample goes through a transposed accumulator with strided writes.
/// - level 0 can come from an 8-bit image (<< 8 fused into the copy), which saves the separate
///   conversion pass and the 16-bit level-0 image.
class PyramidBuilder {
 public:
  using Pyr = ManagedImagePyr<uint16_t>;

  template <typename SrcT>
  void build(Pyr& pyr, const Image<SrcT>& src, size_t num_levels) {
    static_assert(std::is_same<SrcT, uint16_t>::value || std::is_same<SrcT, uint8_t>::value);
    ManagedImage<uint16_t>& image = Access::image_of(pyr);
    const size_t w = src.w;
    const size_t h = src.h;

    Access::orig_w_of(pyr) = w;
    const uint16_t* old_ptr = image.ptr;
    image.Reinitialise(w + w / 2, h);
    if (image.ptr != old_ptr) image.Fill(0);

    Image<uint16_t> l0 = Access::lvl_of(pyr, 0);
    for (size_t y = 0; y < h; y++) {
      if constexpr (std::is_same<SrcT, uint8_t>::value) {
        widen(src.RowPtr(y), l0.RowPtr(y), w);
      } else {
        std::memcpy(l0.RowPtr(y), src.RowPtr(y), w * sizeof(uint16_t));
      }
    }

    for (size_t i = 0; i < num_levels; i++) {
      Image<uint16_t> lp1 = Access::lvl_of(pyr, i + 1);
      subsample(static_cast<const Pyr&>(pyr).lvl(i), lp1);
    }
  }

  /// Same arithmetic as ManagedImagePyr::subsample (kernel 1 4 6 4 1 in both directions, reflect-101 borders,
  /// (sum + 128) >> 8), row by row.
  void subsample(const Image<const uint16_t>& img, Image<uint16_t>& img_sub) {
    const int w = int(img.w);
    const int h = int(img.h);
    const int sw = int(img_sub.w);
    const int sh = int(img_sub.h);

    vsum.resize(w);
    even.resize(w / 2 + 2);
    odd.resize(w / 2 + 2);

    for (int r = 0; r < sh; r++) {
      const uint16_t* row_m2 = img.RowPtr(std::abs(2 * r - 2));
      const uint16_t* row_m1 = img.RowPtr(std::abs(2 * r - 1));
      const uint16_t* row = img.RowPtr(2 * r);
      const uint16_t* row_p1 = img.RowPtr(Pyr::border101(2 * r + 1, h));
      const uint16_t* row_p2 = img.RowPtr(Pyr::border101(2 * r + 2, h));

      int* v = vsum.data();
      vertical(row_m2, row_m1, row, row_p1, row_p2, v, w);

      uint16_t* out = img_sub.RowPtr(r);

      // Columns whose 5 taps are all inside [0, w) without reflection: 2c - 2 >= 0 and 2c + 2 <= w - 1.
      // Split v into even/odd columns so the inner loop is unit-stride: out[c] uses
      // e[c-1] + 4 o[c-1] + 6 e[c] + 4 o[c] + e[c+1] with e[k] = v[2k], o[k] = v[2k + 1].
      int* e = even.data();
      int* o = odd.data();
      deinterleave(v, e, o, w);

      int c_end = (w - 3) / 2 + 1;  // first c with 2c + 2 > w - 1
      if (c_end > sw) c_end = sw;
      int c = 0;
      for (; c < 1 && c < sw; c++) out[c] = horizontal(v, w, c);
      if (c < c_end) {
        horizontalInterior(e, o, out, c, c_end);
        c = c_end;
      }
      for (; c < sw; c++) out[c] = horizontal(v, w, c);
    }
  }

 private:
  // The loops below take __restrict pointers: without them GCC's run-time alias checks can send them to the
  // scalar fallback.
  __attribute__((noinline)) static void widen(const uint8_t* __restrict s, uint16_t* __restrict d, size_t n) {
    for (size_t x = 0; x < n; x++) d[x] = uint16_t(s[x]) << 8;
  }

  __attribute__((noinline)) static void vertical(const uint16_t* __restrict m2, const uint16_t* __restrict m1, const uint16_t* __restrict c0,
                       const uint16_t* __restrict p1, const uint16_t* __restrict p2, int* __restrict v, int n) {
    for (int c = 0; c < n; c++) {
      v[c] = int(m2[c]) + 4 * int(m1[c]) + 6 * int(c0[c]) + 4 * int(p1[c]) + int(p2[c]);
    }
  }

  __attribute__((noinline)) static void deinterleave(const int* __restrict v, int* __restrict e, int* __restrict o, int n) {
    const int no = n / 2;
    for (int k = 0; k < no; k++) {
      e[k] = v[2 * k];
      o[k] = v[2 * k + 1];
    }
    if (n % 2 != 0) e[no] = v[2 * no];
  }

  __attribute__((noinline)) static void horizontalInterior(const int* __restrict e, const int* __restrict o, uint16_t* __restrict out, int c0,
                                 int c1) {
    for (int c = c0; c < c1; c++) {
      const int val = e[c - 1] + 4 * o[c - 1] + 6 * e[c] + 4 * o[c] + e[c + 1];
      out[c] = uint16_t((val + (1 << 7)) >> 8);
    }
  }

  static uint16_t horizontal(const int* v, int w, int c) {
    const int val = v[std::abs(2 * c - 2)] + 4 * v[std::abs(2 * c - 1)] + 6 * v[2 * c] +
                    4 * v[Pyr::border101(2 * c + 1, w)] + v[Pyr::border101(2 * c + 2, w)];
    return uint16_t((val + (1 << 7)) >> 8);
  }

  // ManagedImagePyr keeps its mipmap protected; reach it through a derived class's member pointers.
  struct Access : Pyr {
    static ManagedImage<uint16_t>& image_of(Pyr& p) { return p.*(&Access::image); }
    static size_t& orig_w_of(Pyr& p) { return p.*(&Access::orig_w); }
    static Image<uint16_t> lvl_of(Pyr& p, size_t l) { return (p.*(&Access::lvl_internal))(l); }
  };

  std::vector<int> vsum, even, odd;
};

}  // namespace basalt
