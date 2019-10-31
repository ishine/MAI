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

class StridedSliceTest : public OperatorTest {
};

template<typename T>
void stridedSlice(const std::vector<shape_t>& inputShape, const std::vector<T>& inputData,
        const std::vector<int32>& beginData, const std::vector<int32>& endData, const std::vector<int32>& stridesData,
        int32 beginMask, int32 endMask, int32 shrinkAxisMask,
        const std::vector<shape_t>& checkShape, const std::vector<T>& checkData) {
    StridedSliceParam* param = new StridedSliceParam();
    param->beginMask = beginMask;
    param->endMask = endMask;
    param->shrinkAxisMask = shrinkAxisMask;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(STRIDED_SLICE)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "begin", "end", "strides"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .template addTensor<float>("input", inputShape, inputData)
        .template addTensor<int32>("begin", {static_cast<int32>(beginData.size())}, beginData)
        .template addTensor<int32>("end", {static_cast<int32>(endData.size())}, endData)
        .template addTensor<int32>("strides", {static_cast<int32>(stridesData.size())}, stridesData)
        .template addTensor<float>("output", {}, {})
        .template addTensor<float>("check", checkShape, checkData)
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(StridedSliceTest, Dim3) {
    stridedSlice<float>({2,3,2}, {1,2,3,4,5,6,7,8,9,10,11,12},{1,0,0},{2,3,2},{1,1,1},0,0,0,{1,3,2},{7,8,9,10,11,12});
}

} // namespace Test
} // namespace MAI
