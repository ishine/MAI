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

#include <cmath>
#include "BNConvOptimizer.h"
#include "util/MAIType.h"
#include "NeuralNetwork.h"

namespace MAI {

template<typename T>
void foldBatchnormIntoConv2dByType(NeuralNetwork* network,
        Operator* conv2d, Operator* batchnorm);

template<>
void foldBatchnormIntoConv2dByType<float> (
        NeuralNetwork* network, Operator* conv2d, Operator* batchnorm) {
    Tensor* filter = conv2d->getInputTensor(1);
    float* filterData = filter->mutableData<float>();
    Tensor* scale = batchnorm->getInputTensor(1);
    Tensor* offset = batchnorm->getInputTensor(2);
    Tensor* mean = batchnorm->getInputTensor(3);
    Tensor* var = batchnorm->getInputTensor(4);
    float* scaleData = scale->mutableData<float>();
    float* offsetData = offset->mutableData<float>();
    float* meanData = mean->mutableData<float>();
    float* varData = var->mutableData<float>();
    FusedBatchNormParam* param = reinterpret_cast<FusedBatchNormParam*>(batchnorm->getParam());
    for (shape_t o = 0; o < filter->dimO(); ++o) {
        scaleData[o] = scaleData[o] / std::sqrt(varData[o] + param->epsilon);
    }
    shape_t outerSize = 1;
    shape_t innerSize = 1;
    for (shape_t i = 0; i < filter->dimSize(); ++i) {
        if (i < filter->o()) {
            outerSize *= filter->dim(i);
        } else if (i > filter->o()) {
            innerSize *= filter->dim(i);
        }
    }
    for (shape_t outer = 0; outer < outerSize; ++outer) {
        for (shape_t o = 0; o < filter->dim(filter->o()); ++o) {
            for (shape_t inner = 0; inner < innerSize; ++inner) {
                *filterData = *filterData * scaleData[o];
                filterData++;
            }
        }
    }

    if (conv2d->inputNames().size() == 2) {
        for (shape_t o = 0; o < filter->dimO(); ++o) {
            offsetData[o] = offsetData[o] - scaleData[o] * meanData[o];
        }
        std::string newBiasName = conv2d->name() + "__bias";
        conv2d->addInputName(batchnorm->getInputTensor(2)->name()/*offset*/);
    } else if (conv2d->inputNames().size() == 3) {// This just for test
        // Model should optimize as batch norm append to conv2d which with bias is meanless
        ALOGW("batch norm append to conv2d which with bias is meanless");
        Tensor* bias = conv2d->getInputTensor(2);
        float* biasData = bias->mutableData<float>();
        for (shape_t o = 0; o < bias->dim(0); ++o) {
            biasData[o] = offsetData[o] + scaleData[o] * (biasData[o] - meanData[o]);
        }
        network->removeTensor(offset->name());
    }
    // use output tensor of batchnorm instead of original output tensor of conv2d
    conv2d->replaceOutputName(conv2d->outputName(0), batchnorm->outputName(0));
    network->removeTensor(scale->name());
    network->removeTensor(mean->name());
    network->removeTensor(var->name());
    network->removeOperator(batchnorm->name());
}

void BNConvOptimizer::optimize() {
    std::vector<std::string> opNames = mNeuralNetwork->getOperatorNames();
    for (std::vector<std::string>::size_type i = 0; i < opNames.size(); ++i) {
        std::string& opName = opNames[i];
        Operator* op = mNeuralNetwork->getOperator(opName);
        //TODO:(gavinchen) check group conv2d is ok
        if (op->type() == CONV2D || op->type() == DEPTHWISE_CONV2D) {
            while(i  + 1 < opNames.size()) {
                std::string& nextOpName = opNames[i + 1];
                Operator* nextOp = mNeuralNetwork->getOperator(nextOpName);
                if (nextOp->type() == BIAS_ADD) {
                    i++;
                    foldBiasaddIntoConv2d(op, nextOp);
                } else if (nextOp->type() == FUSED_BATCH_NORM) {
                    i++;
                    foldBatchnormIntoConv2d(op, nextOp);
                } else {
                    break;
                }
            }
        }
    }
}

void BNConvOptimizer::foldBatchnormIntoConv2d(Operator* conv2d, Operator* batchnorm) {
    Tensor* filter = conv2d->getInputTensor(1);
    Tensor* scale = batchnorm->getInputTensor(1);
    if (filter->dataType() != scale->dataType()) {
        MAI_ABORT("Cannot support fold batchnorm(%s) into conv2d(%s)",
                getNameFromDataType(scale->dataType()).c_str(),
                getNameFromDataType(filter->dataType()).c_str());
    }
    if (filter->dataType() == DT_FLOAT) {
        foldBatchnormIntoConv2dByType<float>(mNeuralNetwork, conv2d, batchnorm);
    } else {
        MAI_ABORT("Cannot support fold batchnorm into conv2d with data type(%s)",
                getNameFromDataType(filter->dataType()).c_str());
    }
}

void BNConvOptimizer::foldBiasaddIntoConv2d(
        Operator* conv2d, Operator* biasAdd) {
    if(conv2d->inputNames().size() == 2) {
        conv2d->addInputName(biasAdd->inputName(1));
        conv2d->replaceOutputName(conv2d->outputName(0), biasAdd->outputName(0));
        //TODO:(gavinchen) remove biasadd op
        mNeuralNetwork->removeOperator(biasAdd->name());
    } else if (conv2d->inputNames().size() == 3) { // already have bias, this should never happen
        MAI_ABORT("BiasAdd cannot append to conv2d which has bias already");
    }
}

} // namespace MAI
