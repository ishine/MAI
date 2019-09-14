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

class SplitTest : public OperatorTest {
};

TEST_F(SplitTest, SplitLastDim) {
    SplitParam* param = new SplitParam();
    param->numSplit = 2;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(SPLIT)
            .setDataType(DT_INT32)
            .setInputNames({"input", "axis"})
            .setOutputNames({"output1", "output2"})
            .setParam(param)
            .build())
        .addTensor<int32>("input", {2, 10}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20})
        .addTensor<int32>("axis", {1}, {1})
        .addTensor<int32>("output1", {}, {})
        .addTensor<int32>("output2", {}, {})
        .addTensor<int32>("check1", {2, 5}, {1,2,3,4,5,11,12,13,14,15})
        .addTensor<int32>("check2", {2, 5}, {6,7,8,9,10,16,17,18,19,20})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<int32, int32>(network->getTensor("output1"), network->getTensor("check1"));
    ExpectTensorEQ<int32, int32>(network->getTensor("output2"), network->getTensor("check2"));
}

TEST_F(SplitTest, SplitFirstDim) {
    SplitParam* param = new SplitParam();
    param->numSplit = 2;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(SPLIT)
            .setDataType(DT_INT32)
            .setInputNames({"input", "axis"})
            .setOutputNames({"output1", "output2"})
            .setParam(param)
            .build())
        .addTensor<int32>("input", {2, 10}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20})
        .addTensor<int32>("axis", {1}, {0})
        .addTensor<int32>("output1", {}, {})
        .addTensor<int32>("output2", {}, {})
        .addTensor<int32>("check1", {1, 10}, {1,2,3,4,5,6,7,8,9,10})
        .addTensor<int32>("check2", {1, 10}, {11,12,13,14,15,16,17,18,19,20})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<int32, int32>(network->getTensor("output1"), network->getTensor("check1"));
    ExpectTensorEQ<int32, int32>(network->getTensor("output2"), network->getTensor("check2"));
}

} // namespace Test
} // namespace MAI
