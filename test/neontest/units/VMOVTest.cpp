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

class VMOVTest : public NeonTest {
};

TEST_F(VMOVTest, basic) {
    float s = -2;
    float32x4_t qd = vmovq_n_f32(s);
    printNeonVector(qd);
}

TEST_F(VMOVTest, VMOVN) {
    int16x8_t s = {130,130,130,130,130,130,130,130};
    int8x8_t dd = vmovn_s16(s);
    printNeonVector(dd);
    uint16x8_t s1 = {255,255,255,255,255,255,255,255};
    uint8x8_t dd1 = vmovn_u16(s1);
    printNeonVector(dd1);
}

TEST_F(VMOVTest, VQMOVN) {
    int16x8_t s = {130,130,130,130,130,130,130,130};
    int8x8_t dd = vqmovn_s16(s);
    printNeonVector(dd);
    int8x8_t a = {0, 2, 100, 3};
    int8x8_t b = {0, 2, 1, 3};
    int8x8_t rr = vtbl1_s8(a, b);
    printf("================\n");
    printNeonVector(rr);
}

} // namespace NEON
} // namespace Test
} // namespace MAI
