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
#include "func/Ceil.h"

namespace MAI {
namespace Test {
namespace NEON {

class CeilTest : public NeonTest {
};

TEST_F(CeilTest, basic) {
    float a[8] = {10.1, 8.7, 8.5, 0, -10.5, -1, 3, 3.3};
    const int len = sizeof(a) / sizeof(float);
    float b[len];
    ceil(b, a, len);
    float r[len] = {11, 9, 9, 0, -10, -1, 3, 4};
    ExpectEQ(b, r, len);
}

} // namespace NEON
} // namespace Test
} // namespace MAI
