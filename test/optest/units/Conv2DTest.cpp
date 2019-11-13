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

class Conv2DTest : public OperatorTest {
};

TEST_F(Conv2DTest, floatWithSingleChannelValid_NHWC_HWIO) {
    Conv2DParam* param = new Conv2DParam();
    param->dilations = {1,1,1,1};
    param->strides = {1,1,1,1};
    param->paddingMode = PADDING_VALID;
    param->group = 1;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(CONV2D)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "filter"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input", {2,2,4,1}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16})
        .addTensor<float>("filter", {2,2,1,3}, {1,-1,-1,2,1,-1,3,-1,1,-4,1,1}, HWIO)
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {2,1,3,3}, {
                    -4,2,8,-2,2,8,0,2,8,
                    12,2,8,14,2,8,16,2,8,
                })
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(Conv2DTest, floatWithSingleChannelSame_NHWC_HWIO) {
    Conv2DParam* param = new Conv2DParam();
    param->dilations = {1,1,1,1};
    param->strides = {1,1,1,1};
    param->paddingMode = PADDING_SAME;
    param->group = 1;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(CONV2D)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "filter"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input", {2,2,4,1}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16})
        .addTensor<float>("filter", {2,2,1,3}, {1,-1,-1,2,1,-1,3,-1,1,-4,1,1}, HWIO)
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {2,2,4,3}, {
                    -4, 2, 8,
                    -2, 2, 8,
                    0, 2, 8,
                    28,-12, 4,
                    17, 1, -11,
                    20, 1, -13,
                    23, 1, -15,
                    8, -8, -8,
                    12, 2, 8,
                    14, 2, 8,
                    16, 2, 8,
                    60,-28, 4,
                    41, 1, -27,
                    44, 1, -29,
                    47, 1, -31,
                    16,-16, -16,
                })
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(Conv2DTest, floatWithMultiChannelSame_NHWC_HWIO) {
    Conv2DParam* param = new Conv2DParam();
    param->dilations = {1,1,1,1};
    param->strides = {1,1,1,1};
    param->paddingMode = PADDING_SAME;
    param->group = 1;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(CONV2D)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "filter"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input", {1,2,2,2}, {1,2,3,4,5,6,7,8})
        .addTensor<float>("filter", {1,1,2,2}, {1,2,3,4}, HWIO)
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {1,2,2,2}, {
                    7,10,15,22,
                    23,34,31,46,
                })
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(Conv2DTest, floatWithMultiChannelValid_NHWC_HWIO) {
    Conv2DParam* param = new Conv2DParam();
    param->dilations = {1,1,1,1};
    param->strides = {1,1,1,1};
    param->paddingMode = PADDING_VALID;
    param->group = 1;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(CONV2D)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "filter"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input", {1,2,2,2}, {1,2,3,4,5,6,7,8})
        .addTensor<float>("filter", {1,1,2,2}, {1,2,3,4}, HWIO)
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {1,2,2,2}, {
                    7,10,15,22,
                    23,34,31,46,
                })
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(Conv2DTest, floatWithMultiChannelSameBias_NHWC_HWIO) {
    Conv2DParam* param = new Conv2DParam();
    param->dilations = {1,1,1,1};
    param->strides = {1,1,1,1};
    param->paddingMode = PADDING_SAME;
    param->group = 1;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(CONV2D)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "filter", "bias"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input", {1,2,2,2}, {1,2,3,4,5,6,7,8})
        .addTensor<float>("filter", {1,1,2,2}, {1,2,3,4}, HWIO)
        .addTensor<float>("bias", {2}, {1,2}, HWIO)
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {1,2,2,2}, {
                    8,12,16,24,
                    24,36,32,48,
                })
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

} // namespace Test
} // namespace MAI
