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

#include "NeuralNetwork.h"
#include "core/SimpleNeuralNetwork.h"
#include "core/Allocator.h"

namespace MAI {
namespace Test {

class NetworkBuilder {
public:
    NetworkBuilder() : mNetwork(new SimpleNeuralNetwork()) {
    }
    virtual ~NetworkBuilder() = default;

    NetworkBuilder& addOperator(std::unique_ptr<Operator>&& op) {
        mNetwork->addOperator(op);
        return *this;
    }

    template<typename T>
    NetworkBuilder& addTensor(
            const std::string& name,
            const std::vector<uint64>& dims,
            const std::vector<T>& data,
            const DataFormat dataFormat = NHWC) {
        std::unique_ptr<Tensor> tensor(new Tensor(DataTypeToEnum<T>::value, new CPUAllocator()));
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

    std::unique_ptr<NeuralNetwork> build() {
        return std::move(mNetwork);
    }

private:
    std::unique_ptr<NeuralNetwork> mNetwork;
};

} // namespace Test
} // namespace MAI
