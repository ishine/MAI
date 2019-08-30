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

class ReluTest : public OperatorTest {
};

TEST_F(ReluTest, ReluBasic) {
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(RELU)
            .setDataType(DT_FLOAT)
            .setInputNames({"input"})
            .setOutputNames({"output"})
            .build())
        .addTensor<float>("input", {1, 2, 2, 2}, {0,2,3,4,-7,0,-8,-100})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {1, 2, 2, 2}, {0,2,3,4,0,0,0,0})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

} // namespace Test
} // namespace MAI
