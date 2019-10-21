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

class ConcatTest : public OperatorTest {
};

template<class T>
void cast(const std::vector<shape_t>& input0Shape, int32 axis,
        const std::vector<shape_t>& checkShape, const std::vector<T>& checkData) {
    ConcatParam* param = new ConcatParam();
    param->num = 2;
    param->axis = axis;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(CONCAT)
            .setDataType(DataTypeToEnum<T>::value)
            .setInputNames({"input0", "input1"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .template addTensor<T>("input0", input0Shape, {1,2,3,4,5,6,7,8})
        .template addTensor<T>("input1", {2,2,2,2}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16})
        .template addTensor<T>("output", {}, {})
        .template addTensor<T>("check", checkShape, checkData)
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<T, T>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(ConcatTest, concatAxis3) {
    cast<float>({2,2,2,1}, 3, {2,2,2,3},
                {1, 1, 2,
                 2, 3, 4,
                 3, 5, 6,
                 4, 7, 8,
                 5, 9, 10,
                 6, 11, 12,
                 7, 13, 14,
                 8, 15, 16,});
}

TEST_F(ConcatTest, concatAxisMinus1) {
    cast<float>({2,2,2,1}, -1, {2,2,2,3},
                {1, 1, 2,
                 2, 3, 4,
                 3, 5, 6,
                 4, 7, 8,
                 5, 9, 10,
                 6, 11, 12,
                 7, 13, 14,
                 8, 15, 16,});
}

TEST_F(ConcatTest, concatAxis2) {
    cast<float>({2,2,1,2}, 2, {2,2,3,2},
                {1, 2, 1, 2, 3, 4,
                 3, 4, 5, 6, 7, 8,
                 5, 6, 9, 10, 11, 12,
                 7, 8, 13, 14, 15, 16});
}

TEST_F(ConcatTest, concatAxisMinus2) {
    cast<float>({2,2,1,2}, -2, {2,2,3,2},
                {1, 2, 1, 2, 3, 4,
                 3, 4, 5, 6, 7, 8,
                 5, 6, 9, 10, 11, 12,
                 7, 8, 13, 14, 15, 16});
}

TEST_F(ConcatTest, concatAxis1) {
    cast<float>({2,1,2,2}, 1, {2,3,2,2},
                {1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8,
                 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
}

TEST_F(ConcatTest, concatAxisMinus3) {
    cast<float>({2,1,2,2}, -3, {2,3,2,2},
                {1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8,
                 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
}

TEST_F(ConcatTest, concatAxis0) {
    cast<float>({1,2,2,2}, 0, {3,2,2,2},
                {1, 2, 3, 4, 5, 6, 7, 8,
                 1, 2, 3, 4, 5, 6, 7, 8,
                 9, 10, 11, 12, 13, 14, 15, 16});
}

TEST_F(ConcatTest, concatAxisMinus4) {
    cast<float>({1,2,2,2}, -4, {3,2,2,2},
                {1, 2, 3, 4, 5, 6, 7, 8,
                 1, 2, 3, 4, 5, 6, 7, 8,
                 9, 10, 11, 12, 13, 14, 15, 16});
}
} // namespace Test
} // namespace MAI
