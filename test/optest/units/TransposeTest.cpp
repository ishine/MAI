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

class TransposeTest : public OperatorTest {
};

TEST_F(TransposeTest, transpose3120) {
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(TRANSPOSE)
            .setDataType(DT_INT32)
            .setInputNames({"input", "perm"})
            .setOutputNames({"output"})
            .build())
        .addTensor<int32>("input", {1, 2, 2, 3}, {1,2,3,4,5,6,7,8,9,10,11,12})
        .addTensor<int32>("perm", {4}, {3,1,2,0})
        .addTensor<int32>("output", {}, {})
        .addTensor<int32>("check", {3,2,2,1}, {1,4,7,10,2,5,8,11,3,6,9,12})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<int32, int32>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(TransposeTest, transpose0321) {
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(TRANSPOSE)
            .setDataType(DT_INT32)
            .setInputNames({"input", "perm"})
            .setOutputNames({"output"})
            .build())
        .addTensor<int32>("input", {1, 2, 2, 4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16})
        .addTensor<int32>("perm", {4}, {0,3,2,1})
        .addTensor<int32>("output", {}, {})
        .addTensor<int32>("check", {1,4,2,2}, {1,9,5,13,2,10,6,14,3,11,7,15,4,12,8,16})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<int32, int32>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(TransposeTest, transpose0312) {
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(TRANSPOSE)
            .setDataType(DT_INT32)
            .setInputNames({"input", "perm"})
            .setOutputNames({"output"})
            .build())
        .addTensor<int32>("input", {1, 2, 2, 4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16})
        .addTensor<int32>("perm", {4}, {0,3,1,2})
        .addTensor<int32>("output", {}, {})
        .addTensor<int32>("check", {1,4,2,2}, {1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<int32, int32>(network->getTensor("output"), network->getTensor("check"));
}

} // namespace Test
} // namespace MAI
