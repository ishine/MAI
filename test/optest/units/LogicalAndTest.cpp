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

class LogicalAndTest : public OperatorTest {
};

template<class TI, class TO = int8>
void logicalAndTest(const std::vector<shape_t>& input1Shape, const std::vector<TI>& input1Data,
        const std::vector<shape_t>& input2Shape, const std::vector<TI>& input2Data,
        const std::vector<shape_t>& checkShape, const std::vector<TO>& checkData) {
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(LOGICAL_AND)
            .setDataType(DT_INT8)
            .setInputNames({"input1", "input2"})
            .setOutputNames({"output"})
            .build())
        .template addTensor<TI>("input1", input1Shape, input1Data)
        .template addTensor<TI>("input2", input2Shape, input2Data)
        .template addTensor<TO>("output", {}, {})
        .template addTensor<TO>("check", checkShape, checkData)
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<TO, TO>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(LogicalAndTest, LogicalAndBasic) {
    // no broadcast
    logicalAndTest<int8>({2, 2, 2}, {0, 1, 1, 0, 1, 0, 1, 0},
            {2, 2, 2}, {0, 1, 0, 1, 1, 1, 0, 1},
            {2, 2, 2}, {0, 1, 0, 0, 1, 0, 0, 0});
    // broadcast
    logicalAndTest<int8>({1, 2, 1, 2}, {1, 0, 0, 1},
            {2, 2, 2}, {0, 0, 1, 1, 0, 1, 1, 1},
            {1, 2, 2, 2}, {0, 0, 1, 0, 0, 1, 0, 1});

    // A is scalar
    logicalAndTest<int8>({1}, {1},
            {2, 2, 2}, {1, 0, 0, 1, 0, 1, 1, 1},
            {2, 2, 2}, {1, 0, 0, 1, 0, 1, 1, 1});
    // B is scalar
    logicalAndTest<int8>({2, 1, 2}, {1, 0, 0, 1},
            {1}, {0},
            {2, 1, 2}, {0, 0, 0, 0});
}

} // namespace Test
} // namespace MAI
