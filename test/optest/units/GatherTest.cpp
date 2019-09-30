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

class GatherTest : public OperatorTest {
};

TEST_F(GatherTest, TFGather0) {
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(GATHER)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "indices"})
            .setOutputNames({"output"})
            .build())
        .addTensor<float>("input", {1,3,2,3}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18})
        .addTensor<int32>("indices", {1}, {0})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {1,3,2,3}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(GatherTest, TFGatherDim0And2) {
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(GATHER)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "indices"})
            .setOutputNames({"output"})
            .build())
        .addTensor<float>("input", {3,2,3}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18})
        .addTensor<int32>("indices", {2}, {0,2})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {2,2,3}, {1,2,3,4,5,6,13,14,15,16,17,18})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(GatherTest, TFGatherDimMinus3AndMinus1) {
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(GATHER)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "indices"})
            .setOutputNames({"output"})
            .build())
        .addTensor<float>("input", {3,2,3}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18})
        .addTensor<int32>("indices", {2}, {-3,-1})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {2,2,3}, {1,2,3,4,5,6,13,14,15,16,17,18})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

template<typename T, typename INDICES_TYPE>
static void gatherOnnx(const std::vector<shape_t>& inputShape, const std::vector<T>& inputData,
        const std::vector<shape_t>& indicesShape, const std::vector<INDICES_TYPE>& indicesData,
        const std::vector<shape_t>& checkShape, const std::vector<T>& checkData, int32 axis = 0) {
    GatherParam* param = new GatherParam();
    param->axis = axis;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(GATHER)
            .setDataType(DataTypeToEnum<T>::value)
            .setInputNames({"input", "indices"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .template addTensor<T>("input", inputShape, inputData)
        .template addTensor<INDICES_TYPE>("indices", indicesShape, indicesData)
        .template addTensor<T>("output", {}, {})
        .template addTensor<T>("check", checkShape, checkData)
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(GatherTest, OnnxGatherAxis0) {
    gatherOnnx<float, int32>({3,3,3}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27},
            {2,2}, {0,2,1,2},
            {2,2,3,3}, {
                1,2,3,4,5,6,7,8,9,
                19,20,21,22,23,24,25,26,27,
                10,11,12,13,14,15,16,17,18,
                19,20,21,22,23,24,25,26,27,});
}

TEST_F(GatherTest, OnnxGatherAxis1) {
    gatherOnnx<float, int32>({3,3,3}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27},
            {2,2}, {0,2,1,2},
            {3,2,2,3}, {
                1,2,3,7,8,9,
                4,5,6,7,8,9,
                10,11,12,16,17,18,
                13,14,15,16,17,18,
                19,20,21,25,26,27,
                22,23,24,25,26,27,}, 1);
}

TEST_F(GatherTest, OnnxGatherAxis2) {
    gatherOnnx<float, int32>({3,3,3}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27},
            {2,2}, {0,2,1,2},
            {3,3,2,2}, {
                1,3,2,3,4,6,5,6,7,9,8,9,
                10,12,11,12,13,15,14,15,16,18,17,18,
                19,21,20,21,22,24,23,24,25,27,26,27,
                }, 2);
}


} // namespace Test
} // namespace MAI
