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

class VDUPTest : public NeonTest {
};

TEST_F(VDUPTest, basic) {
    float s = 2;
    float32x4_t qd = vdupq_n_f32(s);
    printNeonVector(qd);
}

TEST_F(VDUPTest, vdup_lane) {
    float32x2_t s = {1, 2};

    float32x4_t qd = vdupq_lane_f32(s, 0);
    printNeonVector(qd);
}

} // namespace NEON
} // namespace Test
} // namespace MAI
