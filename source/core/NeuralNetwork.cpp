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

#include "NeuralNetwork.h"
#include "source/util/MAIType.h"
#include "tools/converter/tensorflow/TensorflowNetwork.h"

namespace MAI {

/*static*/
std::unique_ptr<NeuralNetwork> NeuralNetwork::getNeuralNetwork(
        const NetworkFormat networkFormat, const std::string& modelPath) {
    ALOGI("getNeuralNetwork:%d", networkFormat);
    switch (networkFormat) {
    case TENSORFLOW:
        ALOGI("getNeuralNetwork TENSORFLOW");
        return std::unique_ptr<NeuralNetwork>(new TensorflowNetwork(modelPath));
    case ONNX:
    case MAI:
    default:
        MAI_ABORT("Unsupported network format:%d", networkFormat);
        return NULL;
    }
}

void NeuralNetwork::addOptimizer(std::unique_ptr<Optimizer> optimizer) {
    mOptimizers.emplace_back(std::move(optimizer));
}

void NeuralNetwork::addOptimizer(OptimizerRule rule) {
    mOptimizers.emplace_back(createOptimizer(rule));
}

Optimizer* NeuralNetwork::createOptimizer(OptimizerRule rule) {
    switch(rule) {
    case FOLD_BN_INTO_CONV2D:
        return NULL;
    case FOLD_ACTIVATION_INTO_CONV2D:
        return NULL;
    }
    return NULL;
}

void NeuralNetwork::startOptimize() {
}

} // namespace MAI
