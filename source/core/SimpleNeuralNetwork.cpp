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

#include "source/core/SimpleNeuralNetwork.h"
#include "source/util/Type.h"

namespace MAI {

SimpleNeuralNetwork::SimpleNeuralNetwork() {
}

MAI_STATUS SimpleNeuralNetwork::init() {
    return MAI_SUCCESS;
}

MAI_STATUS SimpleNeuralNetwork::run() {
    for (auto it = mOperators.begin(); it != mOperators.end(); ++it) {
        (*it)->run();
    }
    return MAI_SUCCESS;
}

MAI_STATUS SimpleNeuralNetwork::addOperator(std::unique_ptr<Operator>& op) {
    op->setNeuralNetwork(this);
    mOperators.emplace_back(std::move(op));
    return MAI_SUCCESS;
}

MAI_STATUS SimpleNeuralNetwork::addTensor(std::unique_ptr<Tensor>& tensor) {
    MAI_CHECK(mTensors.find(tensor->name()) == mTensors.end(), "%s has exists", tensor->name().c_str());
    mTensors.emplace(tensor->name(), std::move(tensor));
    return MAI_SUCCESS;
}

Tensor* SimpleNeuralNetwork::getTensor(const std::string& name) {
    return mTensors[name].get();
}
} // namespace MAI
