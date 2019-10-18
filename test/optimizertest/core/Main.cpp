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

#include <stdio.h>

#include <gtest/gtest.h>
#include "core/MAIEnvironment.h"
#include "source/util/CmdParser.h"

namespace MAI {
namespace Test {

int Main(int argc, char** argv) {
    printf("Running main() from %s\n", __FILE__);
    testing::AddGlobalTestEnvironment(new MAI::Test::MAIEnvironment);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

} // namespace Test
} // namespace MAI

int main(int argc, char** argv) {
    return MAI::Test::Main(argc, argv);
}
