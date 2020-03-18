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

class AllTest : public OperatorTest {
};

template<class T>
void testAll(const std::vector<int32>& axes, bool keepDim,
        const std::vector<shape_t>& inputShape, const std::vector<T>& inputData,
        const std::vector<shape_t>& checkShape, const std::vector<T>& checkData) {
    ReduceParam* param = new ReduceParam();
    param->axes = axes;
    param->keepDim = keepDim;

    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(ALL)
            .setDataType(DT_INT8)
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

    ExpectTensorEQ<T, T>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(AllTest, AllBasic) {
    testAll<int8>({}, false, {2,2,3}, {1,0,1,0,1,1,0,1,1,1,1,0}, {1}, {0});
    testAll<int8>({}, false, {2,2,3}, {0,0,0,0,0,0,0,0,0,0,0,0}, {1}, {0});
    testAll<int8>({}, false, {2,2,3}, {1,1,1,1,1,1,1,1,1,1,1,1}, {1}, {1});
    testAll<int8>({0}, false, {2,2,3}, {1,0,1,0,0,1,1,1,0,0,1,0}, {2,3}, {1, 0, 0, 0, 0, 0});
    testAll<int8>({-3}, true, {2,2,3}, {1,0,1,0,0,1,1,1,0,0,1,0}, {1,2,3}, {1, 0, 0, 0, 0, 0});
    testAll<int8>({1}, false, {2,2,3}, {1,0,1,0,0,1,1,1,0,0,1,0}, {2,3}, {0,0,1,0,1,0});
    testAll<int8>({2}, true, {2,2,3}, {1,0,1,0,0,1,1,1,0,0,1,0}, {2,2,1}, {0,0,0,0});
    testAll<int8>({0,1}, false, {2,2,3}, {1,0,1,0,0,1,1,1,0,0,1,0}, {3}, {0,0,0});
    testAll<int8>({2,1}, false, {2,2,3}, {1,0,1,0,0,1,1,1,0,0,1,0}, {2}, {0,0});
    testAll<int8>({2,1}, true, {2,2,3}, {1,0,1,0,0,1,1,1,0,0,1,0}, {2,1,1}, {0,0});
}

} // namespace Test
} // namespace MAI
