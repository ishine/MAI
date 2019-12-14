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
#include "include/Device.h"

namespace MAI {
namespace Test {

class GPUReluTest : public OperatorTest {
};

TEST_F(GPUReluTest, ReluGPUBasic) {
    std::shared_ptr<Device> device = Device::createDevice(DEVICE_GPU);
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder(device)
        .addOperator(OperatorBuilder()
            .setType(RELU)
            .setDataType(DT_FLOAT)
            .setDeviceType(DEVICE_GPU)
            .setInputNames({"input"})
            .setOutputNames({"output"})
            .build())
        .addTensor<float>("input", {1, 2, 2, 2}, {0,2,3,4,-7,0,-8,-100})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {1, 2, 2, 2}, {0,2,3,4,0,0,0,0})
        .build();
    network->init();
    Context context;
    context.setDevice(device.get());
    network->run(&context);

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(GPUReluTest, ReluGPUBasicDataRace) {
    std::shared_ptr<Device> device = Device::createDevice(DEVICE_GPU);
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder(device)
        .addOperator(OperatorBuilder()
            .setType(RELU)
            .setDataType(DT_FLOAT)
            .setDeviceType(DEVICE_GPU)
            .setInputNames({"input"})
            .setOutputNames({"output"})
            .build())
        .addTensor<float>("input", {8}, {1,1,1,1,1,1,1,1})
        .addTensor<float>("output", {}, {})
        .addTensor<float>("check", {8}, {2,2,2,2,2,2,1,1})
        .build();
    network->init();
    Context context;
    context.setDevice(device.get());
    network->run(&context);
    const float* dd = network->getTensor("output")->data<float>();
    ALOGI("0=%f, 1=%f, 2=%f, 3=%f, 4=%f, 5=%f, 6=%f, 7=%f", dd[0], dd[1], dd[2], dd[3], dd[4], dd[5], dd[6], dd[7]);

    ExpectTensorEQ<float, float>(network->getTensor("output"), network->getTensor("check"));
}

} // namespace Test
} // namespace MAI
