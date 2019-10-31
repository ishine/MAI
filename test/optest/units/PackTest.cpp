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

class PackTest : public OperatorTest {
};

template<class T>
void pack(const std::vector<shape_t>& input0Shape, const std::vector<T>& input0Data,
        const std::vector<shape_t>& input1Shape, const std::vector<T>& input1Data,
        int32 axis,
        const std::vector<shape_t>& checkShape, const std::vector<T>& checkData) {
    PackParam* param = new PackParam();
    param->num = 2;
    param->axis = axis;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(PACK)
            .setDataType(DataTypeToEnum<T>::value)
            .setInputNames({"input0", "input1"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .template addTensor<T>("input0", input0Shape, input0Data)
        .template addTensor<T>("input1", input1Shape, input1Data)
        .template addTensor<T>("output", {}, {})
        .template addTensor<T>("check", checkShape, checkData)
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<T, T>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(PackTest, packDim1Axis0) {
    pack<float>({2}, {1, 2}, {2}, {3, 4}, 0, {2,2},
                {1, 2, 3, 4});
}

TEST_F(PackTest, packDim1Axis1) {
    pack<float>({2}, {1, 2}, {2}, {3, 4}, 1, {2,2},
                {1, 3, 2, 4});
}

TEST_F(PackTest, packDim4) {
    pack<float>({2, 1, 2, 2}, {1,2,3,4,5,6,7,8},
            {2,1,2,2}, {9,10,11,12,13,14,15,16}, 0, {2,2,1,2,2},
            {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16});
    pack<float>({2, 1, 2, 2}, {1,2,3,4,5,6,7,8},
            {2,1,2,2}, {9,10,11,12,13,14,15,16}, 1, {2,2,1,2,2},
            {1,2,3,4,9,10,11,12,5,6,7,8,13,14,15,16});
    pack<float>({2, 1, 2, 2}, {1,2,3,4,5,6,7,8},
            {2,1,2,2}, {9,10,11,12,13,14,15,16}, 2, {2,1,2,2,2},
            {1,2,3,4,9,10,11,12,5,6,7,8,13,14,15,16});
    pack<float>({2, 1, 2, 2}, {1,2,3,4,5,6,7,8},
            {2,1,2,2}, {9,10,11,12,13,14,15,16}, 3, {2,1,2,2,2},
            {1,2,9,10,3,4,11,12,5,6,13,14,7,8,15,16});
    pack<float>({2, 1, 2, 2}, {1,2,3,4,5,6,7,8},
            {2,1,2,2}, {9,10,11,12,13,14,15,16}, 4, {2,1,2,2,2},
            {1,9,2,10,3,11,4,12,5,13,6,14,7,15,8,16});

    pack<float>({2, 1, 2, 2}, {1,2,3,4,5,6,7,8},
            {2,1,2,2}, {9,10,11,12,13,14,15,16}, -5, {2,2,1,2,2},
            {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16});
    pack<float>({2, 1, 2, 2}, {1,2,3,4,5,6,7,8},
            {2,1,2,2}, {9,10,11,12,13,14,15,16}, -4, {2,2,1,2,2},
            {1,2,3,4,9,10,11,12,5,6,7,8,13,14,15,16});
    pack<float>({2, 1, 2, 2}, {1,2,3,4,5,6,7,8},
            {2,1,2,2}, {9,10,11,12,13,14,15,16}, -3, {2,1,2,2,2},
            {1,2,3,4,9,10,11,12,5,6,7,8,13,14,15,16});
    pack<float>({2, 1, 2, 2}, {1,2,3,4,5,6,7,8},
            {2,1,2,2}, {9,10,11,12,13,14,15,16}, -2, {2,1,2,2,2},
            {1,2,9,10,3,4,11,12,5,6,13,14,7,8,15,16});
    pack<float>({2, 1, 2, 2}, {1,2,3,4,5,6,7,8},
            {2,1,2,2}, {9,10,11,12,13,14,15,16}, -1, {2,1,2,2,2},
            {1,9,2,10,3,11,4,12,5,13,6,14,7,15,8,16});
}

} // namespace Test
} // namespace MAI
