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

#include <limits>
#include "core/OperatorTest.h"

namespace MAI {
namespace Test {

class DepthwiseConv2dTest : public OperatorTest {
};

TEST_F(DepthwiseConv2dTest, floatWithSingleChannelValid_NHWC_HWIO) {
    DepthwiseConv2dParam* param = new DepthwiseConv2dParam();
    param->dilations = {1,1,1,1};
    param->strides = {1,1,1,1};
    param->paddingMode = VALID;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(DEPTHWISE_CONV2D)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "filter"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input", {2,2,4,1}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16})
        .addTensor<float>("filter", {2,2,1,1}, {1,2,3,-4}, HWIO)
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {2,1,3,1}, {
                    -4,-2,0,12,14,16,
                })
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(DepthwiseConv2dTest, floatWithSingleChannelSame_NHWC_HWIO) {
    DepthwiseConv2dParam* param = new DepthwiseConv2dParam();
    param->dilations = {1,1,1,1};
    param->strides = {1,1,1,1};
    param->paddingMode = SAME;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(DEPTHWISE_CONV2D)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "filter"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input", {2,2,4,1}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16})
        .addTensor<float>("filter", {2,2,1,1}, {1,2,3,-4}, HWIO)
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {2,2,4,1}, {
                    -4, -2, 0, 28,
                    17, 20, 23, 8,
                    12, 14, 16, 60,
                    41, 44, 47, 16,
                })
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(DepthwiseConv2dTest, floatWithMultiChannelSame_NHWC_HWIO) {
    DepthwiseConv2dParam* param = new DepthwiseConv2dParam();
    param->dilations = {1,1,1,1};
    param->strides = {1,1,1,1};
    param->paddingMode = SAME;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(DEPTHWISE_CONV2D)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "filter"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input", {1,2,2,2}, {1,2,3,4,5,6,7,8})
        .addTensor<float>("filter", {1,1,2,1}, {1,3}, HWIO)
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {1,2,2,2}, {
                    1,6,3,12,
                    5,18,7,24,
                })
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(DepthwiseConv2dTest, floatWithMultiChannelValid_NHWC_HWIO) {
    DepthwiseConv2dParam* param = new DepthwiseConv2dParam();
    param->dilations = {1,1,1,1};
    param->strides = {1,1,1,1};
    param->paddingMode = VALID;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(DEPTHWISE_CONV2D)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "filter"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input", {1,2,2,2}, {1,2,3,4,5,6,7,8})
        .addTensor<float>("filter", {1,1,2,1}, {1,3}, HWIO)
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {1,2,2,2}, {
                    1,6,3,12,
                    5,18,7,24,
                })
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(DepthwiseConv2dTest, floatWithMultiChannelSameBias_NHWC_HWIO) {
    DepthwiseConv2dParam* param = new DepthwiseConv2dParam();
    param->dilations = {1,1,1,1};
    param->strides = {1,1,1,1};
    param->paddingMode = SAME;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(DEPTHWISE_CONV2D)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "filter", "bias"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input", {1,2,2,2}, {1,2,3,4,5,6,7,8})
        .addTensor<float>("filter", {1,1,2,1}, {1,3}, HWIO)
        .addTensor<float>("bias", {2}, {1,2}, HWIO)
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {1,2,2,2}, {
                    2,8,4,14,
                    6,20,8,26,
                })
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

} // namespace Test
} // namespace MAI
