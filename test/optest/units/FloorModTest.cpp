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

class FloorModTest : public OperatorTest {
};

template<class T>
void floorModTest(const std::vector<shape_t>& input1Shape, const std::vector<T>& input1Data,
        const std::vector<shape_t>& input2Shape, const std::vector<T>& input2Data,
        const std::vector<shape_t>& checkShape, const std::vector<T>& checkData) {
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(FLOOR_MOD)
            .setDataType(DT_FLOAT)
            .setInputNames({"input1", "input2"})
            .setOutputNames({"output"})
            .build())
        .template addTensor<T>("input1", input1Shape, input1Data)
        .template addTensor<T>("input2", input2Shape, input2Data)
        .template addTensor<T>("output", {}, {})
        .template addTensor<T>("check", checkShape, checkData)
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<T, T>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(FloorModTest, FloorModBasic) {
    // no broadcast
    floorModTest<float>({2, 2, 2}, {0, -1, 2, 100, 10, -100, 10, 17},
            {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8},
            {2, 2, 2}, {0, 1, 2, 0, 0, 2, 3, 1});
    // broadcast
    floorModTest<float>({1, 2, 1, 2}, {1, 2, 3, 4},
            {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8},
            {1, 2, 2, 2}, {0, 0, 1, 2, 3, 4, 3, 4});

    // A is scalar
    floorModTest<float>({1}, {1},
            {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8},
            {2, 2, 2}, {0, 1, 1, 1, 1, 1, 1, 1});
    // B is scalar
    floorModTest<float>({2, 1, 2}, {1, 2, 3, 4},
            {1}, {2},
            {2, 1, 2}, {1, 0, 1, 0});
}

} // namespace Test
} // namespace MAI
