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

class SigmoidTest : public OperatorTest {
};

TEST_F(SigmoidTest, SigmoidBasic) {
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(SIGMOID)
            .setDataType(DT_FLOAT)
            .setInputNames({"input"})
            .setOutputNames({"output"})
            .build())
        .addTensor<float>("input", {1, 3, 3, 1}, {-9,-5.5f,-3,-0.5f,0,1.25,3,7,8})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {1, 3, 3, 1}, {0.00012339458, 0.00407013763, 0.047425866, 0.37754071,
                0.5, 0.77729988, 0.95257413, 0.999089, 0.99966466})
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

} // namespace Test
} // namespace MAI
