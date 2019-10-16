// Copyright 2019 MAI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <arm_neon.h>
#include "include/Type.h"
#include "ref/TransposeRef.h"

namespace MAI {
namespace Op {
namespace CPU {
namespace NEON {

template<typename T, int32 DIMSIZE>
struct Transpose {
    static void transpose(const std::vector<shape_t>& inputShape, const T* input,
            const int32* perm,
            const std::vector<shape_t>& outputShape, T* output, int32 sizeOfElement = sizeof(T));
};

//min = 236 max = 647 avg = 283.51 std = 65.7872 for perf test TransposePerfTest.transpose0312
template<>
struct Transpose<int32, 2> {
    static void transpose(const std::vector<shape_t>& inputShape, const int32* input,
            const int32* perm,
            const std::vector<shape_t>& outputShape, int32* output, int32 sizeOfElement = sizeof(int32)) {
        shape_t height = inputShape[0];
        shape_t width = inputShape[1];
        int h = 0;
        int w = 0;
        for (; h + 3 < height; h += 4) {
            w = 0;
            for (; w + 3 < width; w += 4) {
//#define PFDIST 8
//                //__builtin_prefetch(input + (h + 0) * width + w + PFDIST);
//                //__builtin_prefetch(input + (h + 1) * width + w + PFDIST);
//                //__builtin_prefetch(input + (h + 2) * width + w + PFDIST);
//                //__builtin_prefetch(input + (h + 3) * width + w + PFDIST);
//#undef PFDIST
                int32x4_t i0 = vld1q_s32(input + h * width + w);
                int32x4_t i1 = vld1q_s32(input + (h + 1) * width + w);
                int32x4_t i2 = vld1q_s32(input + (h + 2) * width + w);
                int32x4_t i3 = vld1q_s32(input + (h + 3) * width + w);

                int32x4x2_t z0 = vzipq_s32(i0, i1);
                int32x4x2_t z1 = vzipq_s32(i2, i3);

                vst1q_s32(output + (w + 0) * height + h, vcombine_s32(vget_low_s32(z0.val[0]), vget_low_s32(z1.val[0])));
                vst1q_s32(output + (w + 1) * height + h, vcombine_s32(vget_high_s32(z0.val[0]), vget_high_s32(z1.val[0])));
                vst1q_s32(output + (w + 2) * height + h, vcombine_s32(vget_low_s32(z0.val[1]), vget_low_s32(z1.val[1])));
                vst1q_s32(output + (w + 3) * height + h, vcombine_s32(vget_high_s32(z0.val[1]), vget_high_s32(z1.val[1])));
            }
            for (; w < width; ++w) {
                output[w * height + h] = input[h * width + w];
            }
        }
        for (; h < height; ++h) {
            for (w = 0; w < width; ++w) {
                output[w * height + h] = input[h * width + w];
            }
        }
    }
};

template<>
struct Transpose<float, 2> {
    static void transpose(const std::vector<shape_t>& inputShape, const float* input,
            const int32* perm,
            const std::vector<shape_t>& outputShape, float* output, int32 sizeOfElement = sizeof(float)) {
        shape_t height = inputShape[0];
        shape_t width = inputShape[1];
        shape_t h = 0;
        for (; h + 3 < height; h += 4) {
            shape_t w = 0;
            for (; w + 3 < width; w += 4) {
                float32x4_t i0 = vld1q_f32(input + h * width + w);
                float32x4_t i1 = vld1q_f32(input + (h + 1) * width + w);
                float32x4_t i2 = vld1q_f32(input + (h + 2) * width + w);
                float32x4_t i3 = vld1q_f32(input + (h + 3) * width + w);

                float32x4x2_t z0 = vzipq_f32(i0, i1);
                float32x4x2_t z1 = vzipq_f32(i2, i3);

                vst1q_f32(output + (w + 0) * height + h, vcombine_f32(vget_low_f32(z0.val[0]), vget_low_f32(z1.val[0])));
                vst1q_f32(output + (w + 1) * height + h, vcombine_f32(vget_high_f32(z0.val[0]), vget_high_f32(z1.val[0])));
                vst1q_f32(output + (w + 2) * height + h, vcombine_f32(vget_low_f32(z0.val[1]), vget_low_f32(z1.val[1])));
                vst1q_f32(output + (w + 3) * height + h, vcombine_f32(vget_high_f32(z0.val[1]), vget_high_f32(z1.val[1])));
            }
            for (; w < width; ++w) {
                output[w * height + h] = input[h * width + w];
            }
        }
        for (; h < height; ++h) {
            for (shape_t w = 0; w < width; ++w) {
                output[w * height + h] = input[h * width + w];
            }
        }
    }
};

/**
 * This can receive any dim size array which with any type
 */
template<>
struct Transpose<uint8, MAI_DYNAMIC_DIM> {
    static void transpose(const std::vector<shape_t>& inputShape, const uint8* input,
            const int32* perm,
            const std::vector<shape_t>& outputShape, uint8* output, int32 sizeOfElement) {
        Ref::Transpose<uint8, MAI_DYNAMIC_DIM>::transpose(inputShape, input, perm, outputShape, output, sizeOfElement);
    }
};

} // namespace NEON
} // namespace CPU
} // namespace Op
} // namespace MAI
