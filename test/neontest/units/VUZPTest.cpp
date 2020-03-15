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

class VUZPTest : public NeonTest {
};

TEST_F(VUZPTest, VUZP_s16) {
    int16x4_t a = {1, 2, 3, 4};
    int16x4_t b = {5, 6, 7, 8};
    int16x4x2_t rr = vuzp_s16(a, b);
    printNeonVector(rr.val[0]);
    printNeonVector(rr.val[1]);
}

} // namespace NEON
} // namespace Test
} // namespace MAI
