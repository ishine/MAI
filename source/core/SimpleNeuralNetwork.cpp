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

#include "core/SimpleNeuralNetwork.h"
#include "source/core/OpenMP.h"
#include "Allocator.h"
#include "util/MAIType.h"
#include "source/ops/cpu/CPURegister.h"
#include "tools/profiling/Profiler.h"

namespace MAI {

SimpleNeuralNetwork::SimpleNeuralNetwork() {
    Op::CPU::CPURegister::getInstance();

    OpenMP::setNumThreads(OpenMP::getNumCPUCores());
}

MAI_STATUS SimpleNeuralNetwork::init() {
    for (auto it = mOperators.begin(); it != mOperators.end(); ++it) {
        (*it)->init();
    }
    return MAI_SUCCESS;
}

MAI_STATUS SimpleNeuralNetwork::run() {
#if 0
    const std::string kOutputDir = "output";
    for(int32 i = 0; i < mModelInputs.size(); ++i) {
        getTensor(mModelInputs[i])->toFile(kOutputDir);
    }
    for (auto it = mOperators.begin(); it != mOperators.end(); ++it) {
        ALOGI("run %s", (*it)->name().c_str());
        (*it)->run();
        for (int32 i = 0; i < (*it)->outputNames().size(); ++i) {
            (*it)->getOutputTensor(i)->toFile(kOutputDir);
        }
    }
#else
    for (auto it = mOperators.begin(); it != mOperators.end(); ++it) {
        //ALOGI("run %s", (*it)->name().c_str());
        SCOPED_OPERATOR_PROFILE(getProfiler(), (*it)->name(), getNameFromOperator((*it)->type()));
        (*it)->run();
    }
#endif
    return MAI_SUCCESS;
}

MAI_STATUS SimpleNeuralNetwork::addOperator(std::unique_ptr<Operator>& op) {
    op->setNeuralNetwork(this);
    mOperatorNames.emplace_back(op->name());
    mOperators.emplace_back(std::move(op));
    return MAI_SUCCESS;
}

MAI_STATUS SimpleNeuralNetwork::removeOperator(const std::string& opName) {
    for (auto it = mOperatorNames.begin(); it != mOperatorNames.end(); ++it) {
        if ((*it) == opName) {
            mOperatorNames.erase(it);
            break;
        }
    }

    for (auto it = mOperators.begin(); it != mOperators.end(); ++it) {
        if ((*it)->name() == opName) {
            mOperators.erase(it);
            break;
        }
    }

    return MAI_SUCCESS;
}

MAI_STATUS SimpleNeuralNetwork::addTensor(std::unique_ptr<Tensor>& tensor) {
    MAI_CHECK(mTensors.find(tensor->name()) == mTensors.end(), "%s has exists", tensor->name().c_str());
    mTensorNames.emplace_back(tensor->name());
    mTensors.emplace(tensor->name(), std::move(tensor));
    return MAI_SUCCESS;
}

MAI_STATUS SimpleNeuralNetwork::removeTensor(const std::string& tensorName) {
    for (auto it = mTensorNames.begin(); it != mTensorNames.end(); ++it) {
        if ((*it) == tensorName) {
            mTensorNames.erase(it);
            break;
        }
    }

    for (auto it = mTensors.begin(); it != mTensors.end(); ++it) {
        if (it->first == tensorName) {
            mTensors.erase(it);
            break;
        }
    }

    return MAI_SUCCESS;
}

Tensor* SimpleNeuralNetwork::getTensor(const std::string& name) {
    return mTensors[name].get();
}

std::vector<std::string> SimpleNeuralNetwork::getTensorNames() {
    return mOperatorNames;
}

Operator* SimpleNeuralNetwork::getOperator(const std::string& name) {
    for (auto it = mOperators.begin(); it != mOperators.end(); ++it) {
        if ((*it)->name() == name) {
            return it->get();
        }
    }
    return NULL;
}

std::vector<std::string> SimpleNeuralNetwork::getOperatorNames() {
    return mOperatorNames;
}

std::vector<std::string> SimpleNeuralNetwork::getModelInputs() {
    return mModelInputs;
}

std::vector<std::string> SimpleNeuralNetwork::getModelOutputs() {
    return mModelOutputs;
}

void SimpleNeuralNetwork::addModelInput(const std::string& inputName,
        DataType dataType, DataFormat dataFormat,
        const std::vector<shape_t>& inputShape) {
    mModelInputs.emplace_back(inputName);
    std::unique_ptr<Tensor> tensor(new Tensor(dataType, new CPUAllocator()));
    tensor->setName(inputName);
    tensor->setDataFormat(dataFormat);
    tensor->allocateBuffer(inputShape);
    addTensor(tensor);
}

void SimpleNeuralNetwork::addModelOutput(const std::string& outputName) {
    mModelOutputs.emplace_back(outputName);
}

void SimpleNeuralNetwork::builGraph() {

}

} // namespace MAI
