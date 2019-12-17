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

#pragma once

#include <random>
#include <limits>
#include <functional>
#include "NeuralNetwork.h"
#include "core/SimpleNeuralNetwork.h"
#include "core/Allocator.h"
#include "PerformanceRunner.h"
#include "NeuralNetworkRunner.h"
#include "Device.h"

namespace MAI {
namespace Test {

class NetworkBuilder {
public:
    NetworkBuilder(DeviceType deviceType = DEVICE_CPU)
        : mNetwork(new SimpleNeuralNetwork()) {
        //std::shared_ptr<MAI::Device> device = Device::createDevice(mDeviceType);
        mNetwork->setDevice(Device::createDevice(deviceType));
        mNetwork->init();
    }

    NetworkBuilder(std::shared_ptr<Device> device)
        : mNetwork(new SimpleNeuralNetwork()) {
        mNetwork->setDevice(device);
        mNetwork->init();
    }

    virtual ~NetworkBuilder() = default;

    inline NetworkBuilder& addOperator(std::unique_ptr<Operator>&& op) {
        mNetwork->addOperator(op);
        return *this;
    }

    template<typename T>
    NetworkBuilder& addRandomTensor(
            const std::string& name,
            const std::vector<shape_t>& dims,
            const DataFormat dataFormat = NHWC);

    template<typename T>
    NetworkBuilder& addTensor(
            const std::string& name,
            const std::vector<shape_t>& dims,
            const std::vector<T>& data,
            const DataFormat dataFormat = NHWC) {
        Allocator* all = mNetwork->getDevice()->allocator();
        std::unique_ptr<Tensor> tensor(new Tensor(DataTypeToEnum<T>::value, mNetwork->getDevice()->allocator()));
        tensor->setName(name);
        if (!dims.empty()) {
            tensor->allocateBuffer(dims);
        }
        if (!data.empty()) {
            tensor->copy(reinterpret_cast<const void*>(&data[0]), data.size() * sizeof(T));
        }

        tensor->setDataFormat(dataFormat);

        mNetwork->addTensor(tensor);
        return *this;
    }

    inline std::unique_ptr<NeuralNetwork> build() {
        return std::move(mNetwork);
    }

    inline std::unique_ptr<NeuralNetworkRunner> buildRunner() {
        mNeuralNetworkRunner.reset(new NeuralNetworkRunner(mNetwork));
        return std::move(mNeuralNetworkRunner);
    }

    inline std::unique_ptr<PerformanceRunner> buildPerformanceRunner() {
        mNetwork->init();
        mPerformanceRunner.reset(new PerformanceRunner(mNetwork));
        return std::move(mPerformanceRunner);
    }

private:
    std::unique_ptr<NeuralNetwork> mNetwork;
    std::unique_ptr<NeuralNetworkRunner> mNeuralNetworkRunner;
    std::unique_ptr<PerformanceRunner> mPerformanceRunner;
};

template<>
NetworkBuilder& NetworkBuilder::addRandomTensor<int32>(
        const std::string& name,
        const std::vector<shape_t>& dims,
        const DataFormat dataFormat);

template<>
NetworkBuilder& NetworkBuilder::addRandomTensor<float>(
        const std::string& name,
        const std::vector<shape_t>& dims,
        const DataFormat dataFormat);

} // namespace Test
} // namespace MAI
