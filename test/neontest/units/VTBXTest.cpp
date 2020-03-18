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

class VTBXTest : public NeonTest {
};

TEST_F(VTBXTest, VTBX1) {
    int8x8_t dst = {100, 99, 98, 97, 96, 95, 94, 93};
    int8x8_t a = {1, 5, 4, 0, 7, 2, 4, 3};
    int8x8_t b = {0, 4, 7, 1, 5, 5, 3, 8};
    int8x8_t rr = vtbx1_s8(dst, a, b);
    printNeonVector(rr);
}

//TEST_F(VTBXTest, VTBX2) {
//    int8x8x2_t a = {0,1,2,3,4,5,6,7,
//                    93,94,95,96,97,98,99,100};
//    int8x8_t b = {15, 0, 10, 8, 3, 6, 5, 7};
//    int8x8_t rr = vtbx2_s8(a, b);
//    printNeonVector(rr);
//}
//
//TEST_F(VTBXTest, VTBX3) {
//    int8x8x3_t a = {0,1,2,3,4,5,6,7,
//                    10,11,12,13,14,15,16,17,
//                    93,94,95,96,97,98,99,100};
//    int8x8_t b = {15, 0, 10, 16, 17, 8, 11, 4};
//    int8x8_t rr = vtbl3_s8(a, b);
//    printNeonVector(rr);
//}
//
//TEST_F(VTBXTest, VTBX4) {
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
