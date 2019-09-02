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

class BiasAddTest : public OperatorTest {
};

TEST_F(BiasAddTest, BiasAddNHWC) {
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(BIAS_ADD)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "bias"})
            .setOutputNames({"output"})
            .build())
        .addTensor<float>("input", {1, 2, 2, 3}, {1,2,3,4,5,6,7,8,9,10,11,12}, NHWC)
        .addTensor<float>("bias", {3}, {1,2,3})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {1,2,2,3}, {
                2,4,6,
                5,7,9,
                8,10,12,
                11,13,15,})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(BiasAddTest, BiasAddNCHW) {
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(BIAS_ADD)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "bias"})
            .setOutputNames({"output"})
            .build())
        .addTensor<float>("input", {1, 3, 2, 2}, {1,4,7,10,2,5,8,11,3,6,9,12}, NCHW)
        .addTensor<float>("bias", {3}, {1,2,3})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {1,3,2,2}, {
                2,5,8,11,
                4,7,10,13,
                6,9,12,15,})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));

}

} // namespace Test
} // namespace MAI
