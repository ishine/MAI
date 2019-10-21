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

#include "include/Type.h"
#include "include/Operator.h"
#include "include/Tensor.h"
#include "include/Optimizer.h"

namespace MAI {

namespace Profiling {
class Profiler;
}
class NeuralNetwork {
public:
    enum OptimizerRule {
        FOLD_BN_INTO_CONV2D,
        FOLD_ACTIVATION_INTO_CONV2D,
        CONSTANT_FOLD,
    };

    enum NetworkFormat {
        TENSORFLOW,
        ONNX,
        MAI,
    };

public:
    static std::unique_ptr<NeuralNetwork> getNeuralNetwork(
            const NetworkFormat networkFormat, const std::string& modelPath);

    NeuralNetwork() : mProfiler(NULL) {}
    virtual ~NeuralNetwork() = default;

    inline void setProfiler(Profiling::Profiler* profiler) {
        mProfiler = profiler;
    }

    inline Profiling::Profiler* getProfiler() {
        return mProfiler;
    }
    virtual MAI_STATUS init() = 0;
    virtual MAI_STATUS run() = 0;
    virtual MAI_STATUS addOperator(std::unique_ptr<Operator>& op) = 0;
    virtual MAI_STATUS removeOperator(const std::string& opName) = 0;
    virtual MAI_STATUS addTensor(std::unique_ptr<Tensor>& tensor) = 0;
    virtual Tensor* getTensor(const std::string& name) = 0;
    virtual std::vector<std::string> getTensorNames() = 0;
    virtual MAI_STATUS removeTensor(const std::string& tensorName) = 0;
    virtual Operator* getOperator(const std::string& name) = 0;
    virtual std::vector<std::string> getOperatorNames() = 0;

    virtual void addOptimizer(std::unique_ptr<Optimizer> optimizer);
    virtual void addOptimizer(OptimizerRule rule);
    virtual Optimizer* createOptimizer(OptimizerRule rule);
    virtual void startOptimize();
    virtual void builGraph() = 0;
    virtual std::vector<std::string> getModelInputs() = 0;
    virtual std::vector<std::string> getModelOutputs() = 0;

    virtual void addModelInput(const std::string& inputName,
            DataType dataType, DataFormat dataFormat,
            const std::vector<shape_t>& inputShape) = 0;
    virtual void addModelOutput(const std::string& outputName) = 0;
private:
    std::vector<std::unique_ptr<Optimizer> > mOptimizers;
    Profiling::Profiler* mProfiler;
};

} // namespace MAI
