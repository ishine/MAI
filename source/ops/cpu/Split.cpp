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

#include <cstring>
#include "core/OperatorRegister.h"
#include "util/MAIUtil.h"

namespace MAI {
namespace Op {
namespace CPU {

template<typename T>
class Split : public Operator {
public:
    Split() : mNumSplit(0), mAxis(0), mRunFirst(true) {}
    ~Split() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    void setParam(Param* param) override {
        SplitParam* splitParam = reinterpret_cast<SplitParam*>(param);
        if (splitParam) {
            mNumSplit = splitParam->numSplit;
        }
    }

    MAI_STATUS run() override {
        const Tensor* input = getInputTensor(0);
        const Tensor* axisTensor = getInputTensor(1);

        MAI_OP_RUN_FIRST_START
        MAI_CHECK_NULL(input);
        MAI_CHECK_NULL(axisTensor);
        MAI_CHECK(mNumSplit == outputNames().size(), "NumSplit not equal to output tensors");
        // axis -> [-rank(input), rank(input))
        mAxis = *(axisTensor->data<int32>());
        mAxis = mAxis < 0 ? mAxis + input->dimSize() : mAxis;
        MAI_CHECK(input->dim(mAxis) % mNumSplit == 0, "dim of input cannot be divided by numSplit");
        std::vector<shape_t> outputShape(input->shape());
        outputShape[mAxis] = input->dim(mAxis) / mNumSplit;
        for (int32 i = 0; i < mNumSplit; ++i) {
            Tensor* output = getOutputTensor(i);
            MAI_CHECK_NULL(output);
            output->resize(outputShape);
        }

        mOuterSize = std::accumulate(outputShape.begin(), outputShape.begin() + mAxis, 1,
                std::multiplies<shape_t>());
        mInnerSize = std::accumulate(outputShape.begin() + mAxis + 1, outputShape.end(), 1,
                std::multiplies<shape_t>());
        MAI_OP_RUN_FIRST_END

        const T* inputData = input->data<T>();
        Tensor* output0 = getOutputTensor(0);
        for (shape_t outerIdx = 0; outerIdx < mOuterSize; ++outerIdx) {
            shape_t inputIdx = outerIdx * input->dim(mAxis) * mInnerSize;
            shape_t outputIdx = outerIdx * output0->dim(mAxis) * mInnerSize;
            for (shape_t i = 0; i < mNumSplit; ++i) {
                Tensor* output = getOutputTensor(i);
                MAI_CHECK_NULL(output);
                T* outputData = output->mutableData<T>();
                memcpy(outputData + outputIdx, inputData + inputIdx + i * output->dim(mAxis) * mInnerSize, mInnerSize * output->dim(mAxis) * sizeof(T));
            }
        }
        return MAI_FAILED;
    }
private:
    int32 mNumSplit;
    int32 mAxis;
    shape_t mInnerSize;
    shape_t mOuterSize;
    bool mRunFirst;
};

void registerSplit() {
    MAI_REGISTER_OP((OpContext{.opType=SPLIT,}), int32, Split);
    MAI_REGISTER_OP((OpContext{.opType=SPLIT,}), int64, Split);
    MAI_REGISTER_OP((OpContext{.opType=SPLIT,}), float, Split);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
