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
#include "source/core/SimpleNeuralNetwork.h"
#include "source/core/optimizers/BNConvOptimizer.h"
#include "source/core/optimizers/ConstFoldOptimizer.h"

#ifdef MAI_TENSORFLOW_ENABLED
#include "tools/converter/tensorflow/TensorflowParser.h"
#endif

#ifdef MAI_ONNX_ENABLED
#include "tools/converter/onnx/OnnxParser.h"
#endif

namespace MAI {

/*static*/
std::unique_ptr<NeuralNetwork> NeuralNetwork::getNeuralNetwork(
        const NetworkFormat networkFormat, const std::string& modelPath) {
    switch (networkFormat) {
#ifdef MAI_TENSORFLOW_ENABLED
    case TENSORFLOW: {
        std::unique_ptr<NeuralNetwork> network(new SimpleNeuralNetwork());
        Converter::Tensorflow::TensorflowParser parser(network.get());
        parser.parse(modelPath);
        return network;
    }
#endif
#ifdef MAI_ONNX_ENABLED
    case ONNX: {
        std::unique_ptr<NeuralNetwork> network(new SimpleNeuralNetwork());
        Converter::ONNX::OnnxParser parser(network.get());
        parser.parse(modelPath);
        return network;
    }
#endif
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
        return new BNConvOptimizer(this);
    case FOLD_ACTIVATION_INTO_CONV2D:
        return NULL;
    case CONSTANT_FOLD:
        return new ConstFoldOptimizer(this);
    }
    return NULL;
}

void NeuralNetwork::startOptimize() {
    for (auto it = mOptimizers.begin(); it != mOptimizers.end(); ++it) {
        (*it)->optimize();
    }
}

} // namespace MAI
