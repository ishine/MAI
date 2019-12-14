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

class GemmTest : public OperatorTest {
};

TEST_F(GemmTest, GemmBasic1) {
    GemmParam* param = new GemmParam();
    param->alpha = 1.f;
    param->beta = 1.f;
    param->transA = false;
    param->transB = false;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(GEMM)
            .setDataType(DT_FLOAT)
            .setInputNames({"input0", "input1", "input2"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input0", {4,4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16})
        .addTensor<float>("input1", {4,4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16})
        //.addTensor<float>("input2", {4}, {1,2,3,4})
        .addTensor<float>("input2", {4}, {0,0,0,0})
        .addTensor<float>("output", {}, {})
        //.addTensor<float>("check", {4,4}, {91,102,113,124,203,230,257,284,315,358,401,444,427,486,545,604})
        .addTensor<float>("check", {4,4}, {90,100,110,120,202,228,254,280,314,356,398,440,426,484,542,600})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}
TEST_F(GemmTest, GemmBasic) {
    GemmParam* param = new GemmParam();
    param->alpha = 1.f;
    param->beta = 1.f;
    param->transA = false;
    param->transB = false;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(GEMM)
            .setDataType(DT_FLOAT)
            .setInputNames({"input0", "input1", "input2"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input0", {2,3}, {1,2,3,4,5,6})
        .addTensor<float>("input1", {3,2}, {1,2,3,4,5,6})
        .addTensor<float>("input2", {2}, {1,2})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {2,2}, {23,30,50,66})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(GemmTest, GemmBasicTransA) {
    GemmParam* param = new GemmParam();
    param->alpha = 1.f;
    param->beta = 1.f;
    param->transA = true;
    param->transB = false;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(GEMM)
            .setDataType(DT_FLOAT)
            .setInputNames({"input0", "input1", "input2"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input0", {3,2}, {1,2,3,4,5,6})
        .addTensor<float>("input1", {3,2}, {1,2,3,4,5,6})
        .addTensor<float>("input2", {2}, {1,2})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {2,2}, {36,46,45,58})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(GemmTest, GemmBasicTransB) {
    GemmParam* param = new GemmParam();
    param->alpha = 1.f;
    param->beta = 1.f;
    param->transA = false;
    param->transB = true;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(GEMM)
            .setDataType(DT_FLOAT)
            .setInputNames({"input0", "input1", "input2"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input0", {2,3}, {1,2,3,4,5,6})
        .addTensor<float>("input1", {2,3}, {1,2,3,4,5,6})
        .addTensor<float>("input2", {2}, {1,2})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {2,2}, {15,34,33,79})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(GemmTest, GemmBasicTransATransB) {
    GemmParam* param = new GemmParam();
    param->alpha = 1.f;
    param->beta = 1.f;
    param->transA = true;
    param->transB = true;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(GEMM)
            .setDataType(DT_FLOAT)
            .setInputNames({"input0", "input1", "input2"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .addTensor<float>("input0", {3,2}, {1,2,3,4,5,6})
        .addTensor<float>("input1", {2,3}, {1,2,3,4,5,6})
        .addTensor<float>("input2", {2}, {1,2})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {2,2}, {23,51,29,66})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

} // namespace Test
} // namespace MAI
