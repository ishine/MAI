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

class SoftmaxTest : public OperatorTest {
};

TEST_F(SoftmaxTest, Softmax1DWithDefaultParam) {
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(SOFTMAX)
            .setDataType(DT_FLOAT)
            .setInputNames({"input"})
            .setOutputNames({"output"})
            .build())
        .addTensor<float>("input", {4}, {1.f, 2.f, 3.f, 4.f})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {4}, {0.0320586f, 0.08714432f, 0.23688284f, 0.64391428f})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(SoftmaxTest, Softmax1DWithAxisMinusOne) {
    SoftmaxParam* param = new SoftmaxParam();
    param->axis = -1;
    param->beta = 1.f;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(SOFTMAX)
            .setDataType(DT_FLOAT)
            .setInputNames({"input"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input", {4}, {1.f, 2.f, 3.f, 4.f})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {4}, {0.0320586f, 0.08714432f, 0.23688284f, 0.64391428f})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(SoftmaxTest, Softmax2DWithDefaultParam) {
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(SOFTMAX)
            .setDataType(DT_FLOAT)
            .setInputNames({"input"})
            .setOutputNames({"output"})
            .build())
        .addTensor<float>("input", {2,4}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {2,4}, {0.0320586f, 0.08714432f, 0.23688284f, 0.64391428f,
                0.0320586f, 0.08714432f, 0.23688284f, 0.64391428f})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(SoftmaxTest, DISABLED_Softmax2DWithAxis0) {
    SoftmaxParam* param = new SoftmaxParam();
    param->axis = 0;
    param->beta = 1.f;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(SOFTMAX)
            .setDataType(DT_FLOAT)
            .setInputNames({"input"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input", {2,4}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {2,4}, {0.01798621f, 0.01798621f, 0.01798621f, 0.01798621f,
                0.98201376f, 0.98201376f, 0.98201376f, 0.98201376f})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(SoftmaxTest, Softmax2DWithAxis1) {
    SoftmaxParam* param = new SoftmaxParam();
    param->axis = 1;
    param->beta = 1.f;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(SOFTMAX)
            .setDataType(DT_FLOAT)
            .setInputNames({"input"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input", {2,4}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {2,4}, {0.0320586f, 0.08714432f, 0.23688284f, 0.64391428f,
                0.0320586f, 0.08714432f, 0.23688284f, 0.64391428f})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}
} // namespace Test
} // namespace MAI
