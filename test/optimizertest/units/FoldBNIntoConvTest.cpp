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

class FoldBNIntoConvTest : public OperatorTest {
};

TEST_F(FoldBNIntoConvTest, Conv2DHWIO) {
    Conv2DParam* convParam = new Conv2DParam();
    convParam->dilations = {1,1,1,1};
    convParam->strides = {1,1,1,1};
    convParam->paddingMode = PADDING_VALID;
    convParam->group = 1;

    FusedBatchNormParam* fusedParam = new FusedBatchNormParam();
    fusedParam->epsilon = 0.001f;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setName("conv2d")
            .setType(CONV2D)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "filter"})
            .setOutputNames({"conv_output"})
            .setParam(convParam)
            .build())
        .addOperator(OperatorBuilder()
            .setName("fused_batch_norm")
            .setType(FUSED_BATCH_NORM)
            .setDataType(DT_FLOAT)
            .setInputNames({"conv_output", "scale", "offset", "mean", "var"})
            .setOutputNames({"fused_output"})
            .setParam(fusedParam)
            .build())
        .addTensor<float>("input", {2,2,4,1}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16})
        .addTensor<float>("filter", {2,2,1,3}, {1,-1,-1,2,1,-1,3,-1,1,-4,1,1}, HWIO)
        .addTensor<float>("conv_output", {}, {})
        .addTensor<float>("scale", {3}, {0.5f,0.4f,0.3f})
        .addTensor<float>("offset", {3}, {0.1f,0.2f,0.3f})
        .addTensor<float>("mean", {3}, {0.5f,0.4f,0.3f})
        .addTensor<float>("var", {3}, {0.3f,0.4f,0.5f})
        .addTensor<float>("fused_output", {}, {})
        .addTensor<float>("check", {2,1,3,3}, {
                -4.00109,1.2106664,3.5635715,-2.1783848,1.2106658,3.5635715,-0.35567856,1.2106664, 3.5635715,
                10.580563,1.2106658,3.5635715,12.403265,1.2106668,3.5635715,14.225976,1.2106664,3.5635715})
        .build();
    network->addOptimizer(NeuralNetwork::FOLD_BN_INTO_CONV2D);
    network->startOptimize();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("fused_output"), network->getTensor("check"));
}

TEST_F(FoldBNIntoConvTest, Conv2D_BiasAdd) {
    Conv2DParam* convParam = new Conv2DParam();
    convParam->dilations = {1,1,1,1};
    convParam->strides = {1,1,1,1};
    convParam->paddingMode = PADDING_VALID;
    convParam->group = 1;

    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setName("conv2d")
            .setType(CONV2D)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "filter"})
            .setOutputNames({"conv_output"})
            .setParam(convParam)
            .build())
        .addOperator(OperatorBuilder()
            .setName("bias_add")
            .setType(BIAS_ADD)
            .setDataType(DT_FLOAT)
            .setInputNames({"conv_output", "bias"})
            .setOutputNames({"biasadd_output"})
            .build())
        .addTensor<float>("input", {2,2,4,1}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16})
        .addTensor<float>("filter", {2,2,1,3}, {1,-1,-1,2,1,-1,3,-1,1,-4,1,1}, HWIO)
        .addTensor<float>("conv_output", {}, {})
        .addTensor<float>("bias", {3}, {0.5f,0.4f,0.3f})
        .addTensor<float>("biasadd_output", {}, {})
        .addTensor<float>("check", {2,1,3,3}, {
                -3.5,2.4,8.3,-1.5,2.4,8.3,0.5,2.4,8.3,
                12.5,2.4,8.3,14.5,2.4,8.3,16.5,2.4,8.3})
        .build();
    network->addOptimizer(NeuralNetwork::FOLD_BN_INTO_CONV2D);
    network->startOptimize();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("biasadd_output"), network->getTensor("check"));
}

TEST_F(FoldBNIntoConvTest, Conv2D_BiasAdd_BatchNorm) {
    Conv2DParam* convParam = new Conv2DParam();
    convParam->dilations = {1,1,1,1};
    convParam->strides = {1,1,1,1};
    convParam->paddingMode = PADDING_VALID;
    convParam->group = 1;

    FusedBatchNormParam* fusedParam = new FusedBatchNormParam();
    fusedParam->epsilon = 0.001f;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setName("conv2d")
            .setType(CONV2D)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "filter"})
            .setOutputNames({"conv_output"})
            .setParam(convParam)
            .build())
        .addOperator(OperatorBuilder()
            .setName("bias_add1")
            .setType(BIAS_ADD)
            .setDataType(DT_FLOAT)
            .setInputNames({"conv_output", "bias1"})
            .setOutputNames({"biasadd1_output"})
            .build())
        .addOperator(OperatorBuilder()
            .setName("fused_batch_norm")
            .setType(FUSED_BATCH_NORM)
            .setDataType(DT_FLOAT)
            .setInputNames({"biasadd1_output", "scale", "offset", "mean", "var"})
            .setOutputNames({"fused_output"})
            .setParam(fusedParam)
            .build())
        .addTensor<float>("input", {1,2,4,1}, {1,2,3,4,5,6,7,8})
        .addTensor<float>("filter", {2,2,1,3}, {1,-1,-1,2,1,-1,3,-1,1,-4,1,1}, HWIO)
        .addTensor<float>("conv_output", {}, {})
        .addTensor<float>("bias1", {3}, {0.5f,0.4f,0.3f})
        .addTensor<float>("biasadd1_output", {}, {})
        .addTensor<float>("scale", {3}, {0.5f,0.4f,0.3f})
        .addTensor<float>("offset", {3}, {0.1f,0.2f,0.3f})
        .addTensor<float>("mean", {3}, {0.5f,0.4f,0.3f})
        .addTensor<float>("var", {3}, {0.3f,0.4f,0.5f})
        .addTensor<float>("fused_output", {}, {})
        .addTensor<float>("check", {1,1,3,3}, {
                -3.5454133,1.4633331,3.6907239,-1.7227081,1.4633331,3.6907239,0.099998094,1.4633331,3.6907239})
        .build();
    network->addOptimizer(NeuralNetwork::FOLD_BN_INTO_CONV2D);
    network->startOptimize();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("fused_output"), network->getTensor("check"));
}

TEST_F(FoldBNIntoConvTest, DepthwiseConv2dHWIO) {
    DepthwiseConv2dParam* depthwiseConv2dParam = new DepthwiseConv2dParam();
    depthwiseConv2dParam->dilations = {1,1,1,1};
    depthwiseConv2dParam->strides = {1,1,1,1};
    depthwiseConv2dParam->paddingMode = PADDING_VALID;

    FusedBatchNormParam* fusedParam = new FusedBatchNormParam();
    fusedParam->epsilon = 0.001f;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setName("depthwise_conv2d")
            .setType(DEPTHWISE_CONV2D)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "filter"})
            .setOutputNames({"depthwise_conv_output"})
            .setParam(depthwiseConv2dParam)
            .build())
        .addOperator(OperatorBuilder()
            .setName("fused_batch_norm")
            .setType(FUSED_BATCH_NORM)
            .setDataType(DT_FLOAT)
            .setInputNames({"depthwise_conv_output", "scale", "offset", "mean", "var"})
            .setOutputNames({"fused_output"})
            .setParam(fusedParam)
            .build())
        .addTensor<float>("input", {2,2,4,1}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16})
        .addTensor<float>("filter", {2,2,1,1}, {1,2,3-4}, HWIO)
        .addTensor<float>("depthwise_conv_output", {}, {})
        .addTensor<float>("scale", {3}, {0.5f,0.4f,0.3f})
        .addTensor<float>("offset", {3}, {0.1f,0.2f,0.3f})
        .addTensor<float>("mean", {3}, {0.5f,0.4f,0.3f})
        .addTensor<float>("var", {3}, {0.3f,0.4f,0.5f})
        .addTensor<float>("fused_output", {}, {})
        .addTensor<float>("check", {2,1,3,1}, {
                -0.35567665,1.4670299,3.2897365,14.225976,16.048683,17.871389})
        .build();
    network->addOptimizer(NeuralNetwork::FOLD_BN_INTO_CONV2D);
    network->startOptimize();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("fused_output"), network->getTensor("check"));
}

TEST_F(FoldBNIntoConvTest, DepthwiseConv2d_BiasAdd) {
    DepthwiseConv2dParam* depthwiseConv2dParam = new DepthwiseConv2dParam();
    depthwiseConv2dParam->dilations = {1,1,1,1};
    depthwiseConv2dParam->strides = {1,1,1,1};
    depthwiseConv2dParam->paddingMode = PADDING_VALID;

    FusedBatchNormParam* fusedParam = new FusedBatchNormParam();
    fusedParam->epsilon = 0.001f;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setName("depthwise_conv2d")
            .setType(DEPTHWISE_CONV2D)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "filter"})
            .setOutputNames({"depthwise_conv_output"})
            .setParam(depthwiseConv2dParam)
            .build())
        .addOperator(OperatorBuilder()
            .setName("bias_add")
            .setType(BIAS_ADD)
            .setDataType(DT_FLOAT)
            .setInputNames({"depthwise_conv_output", "bias"})
            .setOutputNames({"biasadd_output"})
            .build())
        .addTensor<float>("input", {2,2,4,1}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16})
        .addTensor<float>("filter", {2,2,1,1}, {1,2,3-4}, HWIO)
        .addTensor<float>("depthwise_conv_output", {}, {})
        .addTensor<float>("bias", {1}, {0.5f})
        .addTensor<float>("biasadd_output", {}, {})
        .addTensor<float>("check", {2,1,3,1}, {
                0.5,2.5,4.5,16.5,18.5,20.5})
        .build();
    network->addOptimizer(NeuralNetwork::FOLD_BN_INTO_CONV2D);
    network->startOptimize();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("biasadd_output"), network->getTensor("check"));
}

} // namespace Test
} // namespace MAI
