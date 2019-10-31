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

class ArgMinTest : public OperatorTest {
};

TEST_F(ArgMinTest, ArgMinNoKeepDim) {
    ArgMinParam* param = new ArgMinParam();
    param->keepDim = false;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(ARG_MIN)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "axis"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input", {2, 2}, {2, 1, 3, 4})
        .addTensor<int32>("axis", {1}, {0})
        .addTensor<int32>("output", {}, {})
        .addTensor<int32>("check", {2}, {0,0})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<int32, int32>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(ArgMinTest, ArgMinKeepDim) {
    ArgMinParam* param = new ArgMinParam();
    param->keepDim = true;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(ARG_MIN)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "axis"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input", {2, 2}, {2, 1, 3, 4})
        .addTensor<int64>("axis", {1}, {1})
        .addTensor<int64>("output", {}, {})
        .addTensor<int64>("check", {2, 1}, {1,0})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<int64, int64>(network->getTensor("output"), network->getTensor("check"));
}

} // namespace Test
} // namespace MAI
