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

class SumTest : public OperatorTest {
};

template<class T>
void testSum(const std::vector<int32>& axes, bool keepDim,
        const std::vector<shape_t>& inputShape, const std::vector<T>& inputData,
        const std::vector<shape_t>& checkShape, const std::vector<T>& checkData) {
    ReduceParam* param = new ReduceParam();
    param->axes = axes;
    param->keepDim = keepDim;

    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(SUM)
            .setDataType(DT_FLOAT)
            .setInputNames({"input"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .template addTensor<T>("input", inputShape, inputData)
        .template addTensor<T>("output", {}, {})
        .template addTensor<T>("check", checkShape, checkData)
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(SumTest, SumBasic) {
    testSum<float>({}, false, {2,2,3}, {1,2,3,4,5,6,7,8,9,10,11,12}, {1}, {78});
    testSum<float>({0}, false, {2,2,3}, {1,2,3,4,5,6,7,8,9,10,11,12}, {2,3}, {8,10,12,14,16,18});
    testSum<float>({-3}, true, {2,2,3}, {1,2,3,4,5,6,7,8,9,10,11,12}, {1, 2,3}, {8,10,12,14,16,18});
    testSum<float>({1}, false, {2,2,3}, {1,2,3,4,5,6,7,8,9,10,11,12}, {2,3}, {5,7,9,17,19,21});
    testSum<float>({2}, true, {2,2,3}, {1,2,3,4,5,6,7,8,9,10,11,12}, {2,2,1}, {6,15,24,33});
    testSum<float>({0,1}, false, {2,2,3}, {1,2,3,4,5,6,7,8,9,10,11,12}, {3}, {22,26,30});
    testSum<float>({2,1}, false, {2,2,3}, {1,2,3,4,5,6,7,8,9,10,11,12}, {2}, {21,57});
    testSum<float>({2,1}, true, {2,2,3}, {1,2,3,4,5,6,7,8,9,10,11,12}, {2,1,1}, {21,57});
}

} // namespace Test
} // namespace MAI
