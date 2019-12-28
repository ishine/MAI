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

class LogicalNotTest : public OperatorTest {
};

template<class TI, class TO = int8>
void logicalNotTest(const std::vector<shape_t>& input1Shape, const std::vector<TI>& input1Data,
        const std::vector<shape_t>& checkShape, const std::vector<TO>& checkData) {
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(LOGICAL_NOT)
            .setDataType(DT_INT8)
            .setInputNames({"input"})
            .setOutputNames({"output"})
            .build())
        .template addTensor<TI>("input", input1Shape, input1Data)
        .template addTensor<TO>("output", {}, {})
        .template addTensor<TO>("check", checkShape, checkData)
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<TO, TO>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(LogicalNotTest, LogicalNotBasic) {
    logicalNotTest<int8>({2, 2, 2}, {0, 1, 1, 0, 1, 0, 1, 0},
            {2, 2, 2}, {1, 0, 0, 1, 0, 1, 0, 1});
}

} // namespace Test
} // namespace MAI
