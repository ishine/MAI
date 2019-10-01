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
class Gather : public Operator {
public:
    Gather() : mAxis(0) {}
    ~Gather() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    void setParam(Param* param) override {
        GatherParam* gatherParam = reinterpret_cast<GatherParam*>(param);
        if (gatherParam) {
            mAxis = gatherParam->axis;
            delete param;
        }
    }

    MAI_STATUS run() override {
        const Tensor* input = getInputTensor(0);
        const Tensor* indices = getInputTensor(1);
        Tensor* output = getOutputTensor(0);
        MAI_CHECK_NULL(input);
        MAI_CHECK_NULL(indices);
        MAI_CHECK_NULL(output);
        // 1. check axis is valid
        MAI_CHECK(mAxis >= -(input->dimSize()) && mAxis < input->dimSize(),
                "Invalid axis(%d), input rank:%d", mAxis, input->dimSize());
        if (mAxis < 0) {
            mAxis += input->dimSize();
        }
        // 2. check indices
        shape_t inputDimAxis = input->dim(mAxis);
        mIndex.resize(indices->elementSize());
        if (indices->dataType() == DT_INT32) {
            const int32* indicesData = indices->data<int32>();
            for (shape_t i = 0; i < indices->elementSize(); ++i) {
                int32 v = indicesData[i];
                int32 dim = static_cast<int32>(input->dim(mAxis));
                MAI_CHECK((v >= -dim && v < dim),
                        "Invalid indices value(%d), input dim:%d", v, dim);
                if (v < 0) {
                    v += input->dim(mAxis);
                }
                mIndex[i] = v;
            }
        } else if (indices->dataType() == DT_INT64) {
            const int64* indicesData = indices->data<int64>();
            for (shape_t i = 0; i < indices->elementSize(); ++i) {
                int32 v = static_cast<int32>(indicesData[i]);
                int32 dim = static_cast<int32>(input->dim(mAxis));
                MAI_CHECK(v >= -dim && v < dim,
                        "Invalid indices value(%d), dim:%d", v, dim);
                if (v < 0) {
                    v += input->dim(mAxis);
                }
                mIndex[i] = v;
            }
        } else {
            MAI_ABORT("Unsupported data type of indices:%s",
                    getNameFromDataType(indices->dataType()).c_str());
        }
        // 3. compute output shape
        // q + (r + 1) : inputDim0, inputDim1, ... indicesDim0,...indicesDim(rank(indices)), inputDim(mAxis + 1),...inputDim(rank(input))
        std::vector<shape_t> outputShape(indices->dimSize() + input->dimSize() - 1);
        for (shape_t i = 0; i < outputShape.size(); ++i) {
            if (i < mAxis) {
                outputShape[i] = input->dim(i);
            } else if (i < mAxis + indices->dimSize()) {
                outputShape[i] = indices->dim(i - mAxis);
            } else {
                outputShape[i] = input->dim(i - indices->dimSize() + 1);
            }
        }
        output->resize(outputShape);

        const T* inputData = input->data<T>();
        T* outputData = output->mutableData<T>();
        //shape_t outerSize = std::accumulate(input->shape().begin(), input->shape().begin() + mAxis, 1, std::multiplies<shape_t>());
        const auto& inputShape = input->shape();
        shape_t outerSize = std::accumulate(inputShape.begin(), inputShape.begin() + mAxis, 1, std::multiplies<shape_t>());
        shape_t innerSize = std::accumulate(inputShape.begin() + mAxis + 1, inputShape.end(), 1, std::multiplies<shape_t>());
        shape_t index = 0;
        shape_t inputOffset = 0;
        shape_t outputOffset = 0;
        for (int32 i = 0; i < outerSize; ++i) {
            for (int32 j = 0; j < mIndex.size(); ++j) {
                index = mIndex[j];
                inputOffset = (i * input->dim(mAxis) + index) * innerSize;
                memcpy(outputData + outputOffset, inputData + inputOffset, innerSize * sizeof(T));
                outputOffset += innerSize;
            }
        }
        return MAI_SUCCESS;
    }
private:
    int32 mAxis;
    std::vector<int32> mIndex;
};

void registerGather() {
    MAI_REGISTER_OP((OpContext{.opType=GATHER,}), float, Gather);
    MAI_REGISTER_OP((OpContext{.opType=GATHER,}), int32, Gather);
    MAI_REGISTER_OP((OpContext{.opType=GATHER,}), int64, Gather);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
