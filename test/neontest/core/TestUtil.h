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

#include <gtest/gtest.h>

namespace MAI {
namespace Test {
namespace NEON {

template<typename T>
inline void ExpectEQ(const T& a, const T& b) {
    EXPECT_EQ(a, b);
}

template<>
inline void ExpectEQ<float>(const float& a, const float& b) {
    EXPECT_FLOAT_EQ(a, b);
}

template<>
inline void ExpectEQ<double>(const double& a, const double& b) {
    EXPECT_DOUBLE_EQ(a, b);
}

template<typename T>
inline void ExpectEQ(const T* a, const T* b, unsigned int len) {
    for (unsigned int i = 0; i < len; ++i) {
        EXPECT_EQ(a[i], b[i]);
    }
}

template<>
inline void ExpectEQ<float>(const float* a, const float* b, unsigned int len) {
    for (unsigned int i = 0; i < len; ++i) {
        EXPECT_FLOAT_EQ(a[i], b[i]);
    }
}

template<>
inline void ExpectEQ<double>(const double* a, const double* b, unsigned int len) {
    for (unsigned int i = 0; i < len; ++i) {
        EXPECT_DOUBLE_EQ(a[i], b[i]);
    }
}

inline void printNeonVector(float32x4_t q) {
    float r[4];
    vst1q_f32(r, q);
    printf("%f, %f, %f, %f\n", r[0], r[1], r[2], r[3]);
}

inline void printNeonVector(float32x2_t d) {
    float r[2];
    vst1_f32(r, d);
    printf("%f, %f\n", r[0], r[1]);
}

inline void printNeonVector(uint8x8_t d) {
    uint8_t r[8];
    vst1_u8(r, d);
    printf("%d, %d, %d, %d, %d, %d, %d, %d\n",
            r[0], r[1], r[2], r[3],
            r[4], r[5], r[6], r[7]);
}

inline void printNeonVector(int8x8_t d) {
    int8_t r[8];
    vst1_s8(r, d);
    printf("%d, %d, %d, %d, %d, %d, %d, %d\n",
            r[0], r[1], r[2], r[3],
            r[4], r[5], r[6], r[7]);
}

inline void printNeonVector(int16x4_t d) {
    int16_t r[4];
    vst1_s16(r, d);
    printf("%d, %d, %d, %d\n", r[0], r[1], r[2], r[3]);
}

inline void printNeonVector(uint32x2_t d) {
    uint32_t r[2];
    vst1_u32(r, d);
    printf("%u, %u\n", r[0], r[1]);
}

inline void printNeonVector(uint32x4_t d) {
    uint32_t r[4];
    vst1q_u32(r, d);
    printf("%u, %u, %u, %u\n", r[0], r[1], r[2], r[3]);
}

} // namespace NEON
} // namespace Test
} // namespace MAI
