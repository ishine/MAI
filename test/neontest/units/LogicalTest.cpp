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

class LogicalTest : public NeonTest {
};

TEST_F(LogicalTest, VTST_u8) {
    uint8x8_t a = {1, 3, 7, 15, 31};
    uint8x8_t b = {2, 2, 7, 17, 31};
    uint8x8_t c = vtst_u8(a, b);
    printNeonVector(c);
}

TEST_F(LogicalTest, VORR_u8) {
    uint8x8_t a = {1, 3, 7, 15, 31};
    uint8x8_t b = {2, 2, 7, 17, 31};
    uint8x8_t c = vorr_u8(a, b);
    printNeonVector(c);
}

TEST_F(LogicalTest, VORN_u8) {
    uint8x8_t a = {1, 3, 7, 15, 31};
    uint8x8_t b = {2, 2, 7, 17, 31};
    uint8x8_t c = vorn_u8(a, b);
    printf("%u %u %u %u \n", (1 | (~2)) << 24 >> 24, (3 | (~2)), (7 | (~7)), (15 | (~17)));
    printf("%u %u %u %u \n", (2 | (~1)), (2 | (~3)), (7 | (~7)), (17 | (~15)));
    printNeonVector(c);
}

} // namespace NEON
} // namespace Test
} // namespace MAI
