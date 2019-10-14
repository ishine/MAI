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

#include "core/OperatorTest.h"

namespace MAI {
namespace Test {

class TransposePerfTest : public OperatorTest {
};

TEST_F(TransposePerfTest, transpose0312) {
    std::unique_ptr<PerformanceRunner> runner = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(TRANSPOSE)
            .setDataType(DT_INT32)
            .setInputNames({"input", "perm"})
            .setOutputNames({"output"})
            .build())
        .addRandomTensor<int32>("input", {1, 224, 224, 4})
        .addTensor<int32>("perm", {4}, {0,3,1,2})
        .addTensor<int32>("output", {}, {})
        .buildPerformanceRunner();
    runner->run();
}
} // namespace Test
} // namespace MAI
