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
#include <vector>
#include <map>
#include "include/NeuralNetwork.h"

namespace MAI {

class TensorflowNetwork : public NeuralNetwork {
public:
    TensorflowNetwork(const std::string& netPath);
    ~TensorflowNetwork() = default;
    virtual MAI_STATUS init();
    virtual MAI_STATUS run();
    virtual MAI_STATUS addOperator(std::unique_ptr<Operator>& op);
    virtual MAI_STATUS addTensor(std::unique_ptr<Tensor>& tensor);
    virtual Tensor* getTensor(const std::string& name);
    virtual Operator* getOperator(const std::string& name);
    virtual void builGraph();
    virtual void addModelInput(const std::string& inputName, DataType dataType, const std::vector<shape_t>& inputShape);
    virtual void addModelOutput(const std::string& outputName);

    inline std::vector<std::string> getModelInputs() {
        return mModelInputs;
    }
    inline std::vector<std::string> getModelOutputs() {
        return mModelOutputs;
    }
private:
    std::vector<std::unique_ptr<Operator> > mOperators;
    std::map<std::string, std::unique_ptr<Tensor> > mTensors;
    std::vector<std::string> mModelInputs;
    std::vector<std::string> mModelOutputs;
};

} // namespace MAI
