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

class ResizeBilinearTest : public OperatorTest {
};

TEST_F(ResizeBilinearTest, MulNoBroadcast) {
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(RESIZE_BILINEAR)
            .setDataType(DT_FLOAT)
            .setInputNames({"input", "size"})
            .setOutputNames({"output"})
            .build())
        .addTensor<float>("input", {1,3,3,3}, {
                83, 66, 75, 255, 240, 249, 64, 29, 25,
                46, 29, 38, 143, 126, 135, 62, 27, 23,
                216, 208, 191, 114, 105, 89, 209, 183, 153
                })
        .addTensor<int32>("size", {2}, {4, 4})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {1,4,4,3}, {
                83, 66, 75, 212, 196.5, 205.5, 159.5, 134.5, 137, 64, 29, 25,
                55.25, 38.25, 47.25, 142.0625, 125.4375, 134.4375, 116.75, 91, 93.5, 62.5, 27.5, 23.5,
                131, 118.5, 114.5, 129.125, 116.25, 112.625, 132, 110.25, 100, 135.5, 105, 88,
                216, 208, 191, 139.5, 130.75, 114.5, 161.5, 144, 121, 209, 183, 153,
                })
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

} // namespace Test
} // namespace MAI
