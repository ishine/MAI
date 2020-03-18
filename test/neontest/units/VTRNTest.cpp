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

class VTRNTest : public NeonTest {
};

TEST_F(VTRNTest, VTRN_f32) {
    float32x2_t a = {1, 2};
    float32x2_t b = {3, 4};
    float32x2x2_t rr = vtrn_f32(a, b);
    printNeonVector(rr.val[0]);
    printNeonVector(rr.val[1]);
}

TEST_F(VTRNTest, VTRN_int8) {
    int8x8_t a = {1, 2, 3, 4, 5, 6, 7, 8};
    int8x8_t b = {8, 7, 6, 5, 4, 3, 2, 1};
    int8x8x2_t rr = vtrn_s8(a, b);
    printNeonVector(rr.val[0]);
    printNeonVector(rr.val[1]);
}

TEST_F(VTRNTest, VTRN_int16) {
    int16x4_t a = {1, 2, 3, 4};
    int16x4_t b = {5, 6, 7, 8};
    int16x4x2_t r = vtrn_s16(a, b);
    printNeonVector(r.val[0]);
    printNeonVector(r.val[1]);
}
//
//TEST_F(VTRNTest, VTRN4) {
//    int8x8x4_t a = {0,1,2,3,4,5,6,7,
//                    10,11,12,13,14,15,16,17,
//                    20,21,22,23,24,25,26,27,
//                    93,94,95,96,97,98,99,100};
//    int8x8_t b = {15, 0, 16, 23, 24, 30, 31, 7};
//    int8x8_t rr = vtbl4_s8(a, b);
//    printNeonVector(rr);
//}

} // namespace NEON
} // namespace Test
} // namespace MAI
