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

#include <memory>
#include "Tensor.h"
#include "Type.h"
#include "Log.h"

namespace MAI {

class NeuralNetwork;
class Operator {
public:
    ~Operator() = default;
    virtual MAI_STATUS init() = 0;
    virtual MAI_STATUS run() = 0;

    void addInputNames(const std::vector<std::string>& names);

    void addOutputNames(const std::vector<std::string>& names);

    std::vector<std::string>& inputNames();

    std::vector<std::string>& outputNames();

    void setNeuralNetwork(NeuralNetwork* network);

    Tensor* getTensor(const std::string& inputName);

    Tensor* getInputTensor(int inputIdx);

    Tensor* getOutputTensor(int outputIdx);

private:
    NeuralNetwork* mNeuralNetwork;
    std::vector<std::string> mInputNames;
    std::vector<std::string> mOutputNames;
};

} // namespace MAI
