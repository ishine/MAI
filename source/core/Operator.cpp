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
#include "Operator.h"
#include "util/MAIUtil.h"

namespace MAI {

void Operator::setType(MAIOperator opType) {
    mOpType = opType;
}

MAIOperator Operator::type() const {
    return mOpType;
}

void Operator::addInputName(const std::string& name) {
    mInputNames.push_back(name);
}

void Operator::addOutputName(const std::string& name) {
    mOutputNames.push_back(name);
}

void Operator::addInputNames(const std::vector<std::string>& names) {
    mInputNames.insert(mInputNames.end(), names.begin(), names.end());
}

void Operator::addOutputNames(const std::vector<std::string>& names) {
    mOutputNames.insert(mOutputNames.end(), names.begin(), names.end());
}

void Operator::replaceInputName(const std::string& oriName, const std::string& dstName) {
    for (auto it = mInputNames.begin(); it != mInputNames.end(); ++it) {
        if ((*it) == oriName) {
            (*it) = dstName;
        }
    }
}

void Operator::replaceOutputName(const std::string& oriName, const std::string& dstName) {
    for (auto it = mOutputNames.begin(); it != mOutputNames.end(); ++it) {
        if ((*it) == oriName) {
            (*it) = dstName;
        }
    }
}

std::vector<std::string> Operator::inputNames() const {
    return mInputNames;
}

std::vector<std::string> Operator::outputNames() const {
    return mOutputNames;
}

std::vector<std::string>& Operator::inputNames() {
    return mInputNames;
}

std::vector<std::string>& Operator::outputNames() {
    return mOutputNames;
}

std::string& Operator::inputName(int i) {
    MAI_CHECK(i < mInputNames.size(),
            "%d overflow as max input size is %d", i, mInputNames.size());
    return mInputNames[i];
}

std::string& Operator::outputName(int i) {
    MAI_CHECK(i < mOutputNames.size(),
            "%d overflow as max output size is %d", i, mOutputNames.size());
    return mOutputNames[i];
}

void Operator::setName(const std::string& name) {
    mName = name;
}

std::string Operator::name() const {
    return mName;
}

void Operator::setNeuralNetwork(NeuralNetwork* network) {
    mNeuralNetwork = network;
}

Tensor* Operator::getTensor(const std::string& inputName) {
    return mNeuralNetwork->getTensor(inputName);
}

Tensor* Operator::getInputTensor(int inputIdx) {
    if (inputIdx >= mInputNames.size()) {
        return NULL;
    }
    return mNeuralNetwork->getTensor(mInputNames[inputIdx]);
}

Tensor* Operator::getOutputTensor(int outputIdx) {
    if (outputIdx >= mOutputNames.size()) {
        ALOGE("getOutputTensor %d out of index(%d)", outputIdx, mOutputNames.size());
        return NULL;
    }
    return mNeuralNetwork->getTensor(mOutputNames[outputIdx]);
}

void Operator::setParam(Param* param) {
    MAI_UNUSED(param);
    // do nothing
}

Param* Operator::getParam() {
    // implement be sub class
    return NULL;
}

} // namespace MAI
