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

class FillTest : public OperatorTest {
};

TEST_F(FillTest, FillInt32) {
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(FILL)
            .setDataType(DT_INT32)
            .setInputNames({"dims", "value"})
            .setOutputNames({"output"})
            .build())
        .addTensor<int32>("dims", {2}, {2, 3})
        .addTensor<int32>("value", {1}, {9})
        .addTensor<int32>("output", {}, {})
        .addTensor<int32>("check", {2, 3}, {9,9,9,9,9,9})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<int32, int32>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(FillTest, FillFloat) {
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(FILL)
            .setDataType(DT_FLOAT) //data type of value
            .setInputNames({"dims", "value"})
            .setOutputNames({"output"})
            .build())
        .addTensor<int32>("dims", {2}, {2, 3})
        .addTensor<float>("value", {1}, {1.1f})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {2, 3}, {1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

} // namespace Test
} // namespace MAI
