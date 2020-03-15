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

class VEXTTest : public NeonTest {
};

TEST_F(VEXTTest, basic) {
    float32x4_t dm = {1, 2, 3, 4};
    float32x4_t dn = {5, 6, 7, 8};
    float32x4_t dd = vextq_f32(dn, dm, 3);
    printNeonVector(dd);
}

} // namespace NEON
} // namespace Test
} // namespace MAI
