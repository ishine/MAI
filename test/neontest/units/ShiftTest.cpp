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

#include "core/NeonTest.h"

namespace MAI {
namespace Test {
namespace NEON {

class ShiftTest : public NeonTest {
};

TEST_F(ShiftTest, VSRI_N_s8) {
    int8x8_t a = {5, 6, 7, 8};
    int8x8_t b = {64, 96, 112, 120};
    int8x8_t rr = vsri_n_s8(b, a, 2);
    printNeonVector(rr);

    uint8x8_t a0 = {0, 0, 0, 0, 0, 0, 0, 0};
    uint8x8_t a1 = {255, 16, 16, 16, 16, 16, 16, 16};
    uint16x8_t t = vsubl_u8(a0, a1);
    int16x8_t t1 = vreinterpretq_s16_u16(t);
    int8x8_t r = vmovn_s16(t1);
    printNeonVector(r);

    float32x2_t f0 = {1, 1};
    float32x2_t f1 = {1, 0};
    uint32x2_t fe = vceq_f32(f0, f1);
    printNeonVector(fe);
}

} // namespace NEON
} // namespace Test
} // namespace MAI
