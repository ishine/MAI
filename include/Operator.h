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

struct Param {
};

class NeuralNetwork;
class Operator {
public:
    ~Operator() = default;
    virtual MAI_STATUS init() = 0;
    virtual MAI_STATUS run() = 0;

    void addInputName(const std::string& name);

    void addOutputName(const std::string& name);

    void addInputNames(const std::vector<std::string>& names);

    void addOutputNames(const std::vector<std::string>& names);

    void setName(const std::string& name);

    std::string name() const;
    void setType(MAIOperator opType);
    MAIOperator type() const;

    std::vector<std::string>& inputNames();

    std::vector<std::string>& outputNames();

    void setNeuralNetwork(NeuralNetwork* network);

    Tensor* getTensor(const std::string& inputName);

    Tensor* getInputTensor(int inputIdx);

    Tensor* getOutputTensor(int outputIdx);

    virtual void setParam(Param* param);
private:
    NeuralNetwork* mNeuralNetwork;
    std::vector<std::string> mInputNames;
    std::vector<std::string> mOutputNames;
    std::string mName;
    MAIOperator mOpType;
};

struct SqueezeParam : public Param {
public:
    std::vector<int32> squeezeDims;
};

struct PadParam : public Param {
public:
    float constantValue;
    std::vector<int32> paddings;//dim0_begin, dim0_end, ...
};

struct SoftmaxParam : public Param {
public:
    float beta; // default is 1.;
    int32 axis; // default is -1 or 0;
};

struct FusedBatchNormParam : public Param {
public:
    float epsilon;
};

struct ExpandDimsParam : public Param {
public:
    std::vector<int32> axes;
};

struct SplitParam : public Param {
public:
    int32 numSplit;
};

struct Conv2DParam : public Param {
public:
    std::vector<int32> dilations;//4-d TOP-BOTTON-LEFT-RIGHT
    std::vector<int32> strides;//4-d format associated with input format(NHWC or NCHW)
    std::vector<int32> paddings;//4-d TOP-BOTTON-LEFT-RIGHT
    PaddingMode paddingMode;
};

struct DepthwiseConv2dParam : public Conv2DParam {
public:
    //std::vector<int32> dilations;//4-d TOP-BOTTON-LEFT-RIGHT
    //std::vector<int32> strides;//4-d format associated with input format(NHWC or NCHW)
    //std::vector<int32> paddings;//4-d TOP-BOTTON-LEFT-RIGHT
    //PaddingMode paddingMode;
};

struct PoolParam : public Param {
public:
    std::vector<int32> kernelSizes;//4-d format associated with input format(NHWC or NCHW)
    std::vector<int32> strides;//4-d format associated with input format(NHWC or NCHW)
    std::vector<int32> paddings;//4-d TOP-BOTTON-LEFT-RIGHT
    PaddingMode paddingMode;
};

struct ConcatParam : public Param {
public:
    int32 num;
    int32 axis;//[-rank, rank - 1]
};

struct GemmParam : public Param {
public:
    float alpha;
    float beta;
    bool transA;
    bool transB;
};

struct GatherParam : public Param {
public:
    int32 axis;
};

} // namespace MAI
