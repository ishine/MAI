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

class FusedBatchNormTest : public OperatorTest {
};

TEST_F(FusedBatchNormTest, FusedBatchNormTestNHWC) {
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(FUSED_BATCH_NORM)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "scale", "offset", "mean", "variance"})
            .setOutputNames({"output"})
            .build())
        .addTensor<float>("input", {1,2,2,3}, {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f,9.f,10.f,11.f,12.f}, NHWC)
        .addTensor<float>("scale", {3}, {0.5f,0.4f,0.3f})
        .addTensor<float>("offset", {3}, {0.1f,0.2f,0.3f})
        .addTensor<float>("mean", {3}, {0.5f,0.4f,0.3f})
        .addTensor<float>("variance", {3}, {0.3f,0.4f,0.5f})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {1,2,2,3}, {
                0.55567664f, 1.21066642f, 1.44436932f,
                3.28973651f, 3.10566568f, 2.71589041f,
                6.02379608f, 5.00066519f, 3.98741198f,
                8.75785637f, 6.89566469f, 5.25893354f,})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(FusedBatchNormTest, FusedBatchNormTestNHWCEpsilon) {
    FusedBatchNormParam* param = new FusedBatchNormParam();
    param->epsilon = 0.002f;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(FUSED_BATCH_NORM)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "scale", "offset", "mean", "variance"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input", {1,2,2,3}, {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f,9.f,10.f,11.f,12.f}, NHWC)
        .addTensor<float>("scale", {3}, {0.5f,0.4f,0.3f})
        .addTensor<float>("offset", {3}, {0.1f,0.2f,0.3f})
        .addTensor<float>("mean", {3}, {0.5f,0.4f,0.3f})
        .addTensor<float>("variance", {3}, {0.3f,0.4f,0.5f})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {1,2,2,3}, {
                0.55492157f, 1.20940852f, 1.44322896f,
                3.28445101f, 3.10204935f, 2.7134831f,
                6.01398039f, 4.99468994f, 3.98373747f,
                8.74351025f, 6.88733101f, 5.25399208f,})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(FusedBatchNormTest, DISABLED_FusedBatchNormTestNCHW) {
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(FUSED_BATCH_NORM)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "scale", "offset", "mean", "variance"})
            .setOutputNames({"output"})
            .build())
        .addTensor<float>("input", {1,3,2,2}, {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f,9.f,10.f,11.f,12.f}, NCHW)
        .addTensor<float>("scale", {3}, {0.5f,0.4f,0.3f})
        .addTensor<float>("offset", {3}, {0.1f,0.2f,0.3f})
        .addTensor<float>("mean", {3}, {0.5f,0.4f,0.3f})
        .addTensor<float>("variance", {3}, {0.3f,0.4f,0.5f})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {1,2,2,3}, {
                0.55567664f, 1.21066642f, 1.44436932f,
                3.28973651f, 3.10566568f, 2.71589041f,
                6.02379608f, 5.00066519f, 3.98741198f,
                8.75785637f, 6.89566469f, 5.25893354f,})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

} // namespace Test
} // namespace MAI
