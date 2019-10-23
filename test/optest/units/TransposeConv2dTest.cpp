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

class TransposeConv2dTest : public OperatorTest {
};

TEST_F(TransposeConv2dTest, floatWithSingleChannelSame_NHWC_OHWI) {
    TransposeConv2dParam* param = new TransposeConv2dParam();
    param->dilations = {1,1,1,1};
    param->strides = {1,1,1,1};
    param->paddingMode = SAME;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(TRANSPOSE_CONV2D)
            .setDataType(DT_FLOAT)
            .setInputNames({"outputShape", "filter", "input"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<int32>("outputShape", {4}, {1,4,4,1})
        .addTensor<float>("filter", {3,3,1,1}, {1,2,3,4,5,6,7,8,9}, HWOI)
        .addTensor<float>("input", {1,4,4,1}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {1,4,4,1}, {
                29,62,83,75,99,192,237,198,207,372,417,330,263,446,485,365,
                })
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

} // namespace Test
} // namespace MAI
