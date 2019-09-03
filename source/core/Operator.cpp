#include "NeuralNetwork.h"
#include "Operator.h"
#include "util/MAIUtil.h"

namespace MAI {

void Operator::addInputNames(const std::vector<std::string>& names) {
    mInputNames.insert(mInputNames.end(), names.begin(), names.end());
}

void Operator::addOutputNames(const std::vector<std::string>& names) {
    mOutputNames.insert(mOutputNames.end(), names.begin(), names.end());
}

std::vector<std::string>& Operator::inputNames() {
    return mInputNames;
}

std::vector<std::string>& Operator::outputNames() {
    return mOutputNames;
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
    return mNeuralNetwork->getTensor(mInputNames[inputIdx]);
}

Tensor* Operator::getOutputTensor(int outputIdx) {
    return mNeuralNetwork->getTensor(mOutputNames[outputIdx]);
}

void Operator::setParam(Param* param) {
    MAI_UNUSED(param);
    // do nothing
}

} // namespace MAI
