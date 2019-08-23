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

class SqueezeTest : public OperatorTest {
};

TEST_F(SqueezeTest, SqueezeBasic) {
    SqueezeParam* param = new SqueezeParam();
    param->squeezeDims = {3};
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(SQUEEZE)
            .setDataType(DT_FLOAT)
            .setInputNames({"input"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input", {1, 2, 2, 1}, {1.f, 2.f, 3.f, 4.f})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {1, 2, 2}, {1.f, 2.f, 3.f, 4.f})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(SqueezeTest, SqueezeNoDims) {
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(SQUEEZE)
            .setDataType(DT_FLOAT)
            .setInputNames({"input"})
            .setOutputNames({"output"})
            .build())
        .addTensor<float>("input", {1, 2, 2, 1}, {1.f, 2.f, 3.f, 4.f})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {2, 2}, {1.f, 2.f, 3.f, 4.f})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(SqueezeTest, SqueezeNegativeDim) {
    SqueezeParam* param = new SqueezeParam();
    param->squeezeDims = {-4};
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(SQUEEZE)
            .setDataType(DT_FLOAT)
            .setInputNames({"input"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input", {1, 2, 2, 1}, {1.f, 2.f, 3.f, 4.f})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {2, 2, 1}, {1.f, 2.f, 3.f, 4.f})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}
} // namespace Test
} // namespace MAI
