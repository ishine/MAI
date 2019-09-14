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

class ExpandDimsTest : public OperatorTest {
};

static void expandDims(const std::vector<shape_t>& inputDims,
        const std::vector<shape_t>& outputDims,
        const std::vector<int32>& data, int32 axis) {
    ExpandDimsParam* param = new ExpandDimsParam();
    param->axis = axis;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(EXPAND_DIMS)
            .setDataType(DT_INT32)
            .setInputNames({"input"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<int32>("input", inputDims, data)
        .addTensor<int32>("output", {}, {})
        .addTensor<int32>("check", outputDims, data)
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<int32, int32>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(ExpandDimsTest, MultiDims) {
    expandDims({2,3,2}, {1,2,3,2}, {1,2,3,4,5,6,7,8,9,10,11,12}, 0);
    expandDims({2,3,2}, {2,1,3,2}, {1,2,3,4,5,6,7,8,9,10,11,12}, 1);
    expandDims({2,3,2}, {2,3,1,2}, {1,2,3,4,5,6,7,8,9,10,11,12}, 2);
    expandDims({2,3,2}, {2,3,2,1}, {1,2,3,4,5,6,7,8,9,10,11,12}, 3);
    expandDims({2,3,2}, {2,3,2,1}, {1,2,3,4,5,6,7,8,9,10,11,12}, -1);
    expandDims({2,3,2}, {2,3,1,2}, {1,2,3,4,5,6,7,8,9,10,11,12}, -2);
    expandDims({2,3,2}, {2,1,3,2}, {1,2,3,4,5,6,7,8,9,10,11,12}, -3);
    expandDims({2,3,2}, {1,2,3,2}, {1,2,3,4,5,6,7,8,9,10,11,12}, -4);
}

TEST_F(ExpandDimsTest, SingleDims) {
    expandDims({2}, {1,2}, {3,2}, 0);
    expandDims({2}, {2,1}, {3,2}, 1);
    expandDims({2}, {2,1}, {3,2}, -1);
    expandDims({2}, {1,2}, {3,2}, -2);
}

} // namespace Test
} // namespace MAI
