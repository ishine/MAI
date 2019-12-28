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

class GreaterTest : public OperatorTest {
};

template<class TI, class TO = int8>
void greaterTest(const std::vector<shape_t>& input1Shape, const std::vector<TI>& input1Data,
        const std::vector<shape_t>& input2Shape, const std::vector<TI>& input2Data,
        const std::vector<shape_t>& checkShape, const std::vector<TO>& checkData) {
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(GREATER)
            .setDataType(DT_FLOAT)
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

TEST_F(GreaterTest, GreaterBasic) {
    // no broadcast
    greaterTest<float>({2, 2, 2}, {1, 4, 1, 0, 8, 100, -100, 2000},
            {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8},
            {2, 2, 2}, {0, 1, 0, 0, 1, 1, 0, 1});
    // broadcast
    greaterTest<float>({1, 2, 1, 2}, {-1, 0, 5, 100},
            {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8},
            {1, 2, 2, 2}, {0, 0, 0, 0, 0, 1, 0, 1});

    // A is scalar
    greaterTest<float>({1}, {5},
            {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8},
            {2, 2, 2}, {1, 1, 1, 1, 0, 0, 0, 0});
    // B is scalar
    greaterTest<float>({2, 1, 2}, {1, 2, 3, 4},
            {1}, {2},
            {2, 1, 2}, {0, 0, 1, 1});
}

} // namespace Test
} // namespace MAI
