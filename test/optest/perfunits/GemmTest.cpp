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

class GemmPerfTest : public OperatorTest {
};

TEST_F(GemmPerfTest, gemm_1024_1024_1024) {
    GemmParam* param = new GemmParam();
    param->alpha = 1.f;
    param->beta = 1.f;
    param->transA = false;
    param->transB = false;
    std::unique_ptr<PerformanceRunner> runner = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(GEMM)
            .setDataType(DT_FLOAT)
            .setInputNames({"inputA", "inputB", "inputC"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addRandomTensor<float>("inputA", {1024, 1024})
        .addRandomTensor<float>("inputB", {1024, 1024})
        .addRandomTensor<float>("inputC", {1024})
        .addTensor<float>("output", {}, {})
        .buildPerformanceRunner();
    runner->run();
}

TEST_F(GemmPerfTest, gemm_400_400_400) {
    GemmParam* param = new GemmParam();
    param->alpha = 1.f;
    param->beta = 1.f;
    param->transA = false;
    param->transB = false;
    std::unique_ptr<PerformanceRunner> runner = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(GEMM)
            .setDataType(DT_FLOAT)
            .setInputNames({"inputA", "inputB", "inputC"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addRandomTensor<float>("inputA", {400, 400})
        .addRandomTensor<float>("inputB", {400, 400})
        .addRandomTensor<float>("inputC", {400})
        .addTensor<float>("output", {}, {})
        .buildPerformanceRunner();
    runner->run();
}


} // namespace Test
} // namespace MAI
