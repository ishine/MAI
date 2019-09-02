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

class DISABLED_PadTest : public OperatorTest {
};

TEST_F(DISABLED_PadTest, PadHW) {
    PadParam* param = new PadParam();
    param->paddings = {0,0,1,1,1,1,0,0};

    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(PAD)
            .setDataType(DT_FLOAT)
            .setInputNames({"input"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input", {1, 3, 3, 1}, {1,2,3,4,5,6,7,8,9})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {1, 5, 5, 1}, {
                0,0,0,0,0,
                0,1,2,3,0,
                0,4,5,6,0,
                0,7,8,9,0,
                0,0,0,0,0,})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(DISABLED_PadTest, PadChannel) {
    PadParam* param = new PadParam();
    param->paddings = {0,0,0,0,0,0,1,2};

    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(PAD)
            .setDataType(DT_FLOAT)
            .setInputNames({"input"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input", {1, 2, 2, 2}, {1,2,3,4,5,6,7,8})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {1, 2, 2, 5}, {
                0,1,2,0,0,
                0,3,4,0,0,
                0,5,6,0,0,
                0,7,8,0,0,})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(DISABLED_PadTest, PadBatch) {
    PadParam* param = new PadParam();
    param->paddings = {1,1,0,0,0,0,0,0};

    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(PAD)
            .setDataType(DT_FLOAT)
            .setInputNames({"input"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input", {1, 3, 3, 1}, {1,2,3,4,5,6,7,8,9})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {3, 3, 3, 1}, {
                0,0,0,
                0,0,0,
                0,0,0,
                1,2,3,
                4,5,6,
                7,8,9,
                0,0,0,
                0,0,0,
                0,0,0})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}


TEST_F(DISABLED_PadTest, PadAll) {
    PadParam* param = new PadParam();
    param->paddings = {1,1,1,1,1,1,1,1};

    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(PAD)
            .setDataType(DT_FLOAT)
            .setInputNames({"input"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input", {1, 2, 2, 1}, {1,2,3,4})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {3, 4, 4, 3}, {
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,

                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,

                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,

                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,

                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,

                0,0,0,
                0,1,0,
                0,2,0,
                0,0,0,

                0,0,0,
                0,3,0,
                0,4,0,
                0,0,0,

                0,0,0,
                0,0,0,
                0,0,0,

                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,

                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,

                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,

                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

} // namespace Test
} // namespace MAI
