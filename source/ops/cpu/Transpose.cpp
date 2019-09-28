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

#include "core/OperatorRegister.h"
#include "util/MAIUtil.h"

namespace MAI {
namespace Op {
namespace CPU {

template<typename T>
class Transpose : public Operator {
public:
    Transpose() = default;
    ~Transpose() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    MAI_STATUS run() override {
        const Tensor* input = getInputTensor(INPUT);
        const Tensor* perm = getInputTensor(PERM);
        Tensor* output = getOutputTensor(OUTPUT);
        // run first
        MAI_CHECK_NULL(input);
        MAI_CHECK_NULL(perm);
        MAI_CHECK_NULL(output);
        MAI_CHECK(input->dimSize() == perm->elementSize(), "rank of input must be equal to perm data size");
        const int32* permData = perm->data<int32>();
        std::vector<shape_t> outputShape(input->dimSize());
        std::vector<shape_t> inputStrides(input->dimSize());
        shape_t gap = 1;
        for (int32 i = input->dimSize() - 1; i >= 0; --i) {
            inputStrides[i] = gap;
            gap *= input->dim(i);
        }
        mStrides.resize(input->dimSize());
        for (shape_t i = 0; i < input->dimSize(); ++i) {
            outputShape[i] = input->dim(permData[i]);
            mStrides[i] = inputStrides[permData[i]];
        }
        output->resize(outputShape);
        mOutputStrides.resize(input->dimSize());
        gap = 1;
        for (int32 i = output->dimSize() - 1; i >= 0; --i) {
            mOutputStrides[i] = gap;
            gap *= output->dim(i);
        }
        // run first end
        const T* inputData = input->data<T>();
        T* outputData = output->mutableData<T>();
        shape_t loopCount = 1;
        std::vector<shape_t> indexes(output->dimSize(), 0);
        for (int32 i = output->dimSize() - 1; i >= 0; --i) {
            for (shape_t k = 0; k < i; ++k) {
                indexes[k] = 0;
            }
            shape_t innerSize = 1;
            if (i == output->dimSize() - 1) {
                innerSize = 0;
            } else {
                for (shape_t k = i + 1; k < output->dimSize(); ++k) {
                    innerSize *= output->dim(k);
                }
            }
            loopCount *= output->dim(i);
            for (shape_t j = innerSize; j < loopCount; ++j) {
                shape_t tmpLoopCount = j;
                for(shape_t k = i; k < output->dimSize(); ++k) {
                    indexes[k] = tmpLoopCount / mOutputStrides[k];
                    tmpLoopCount %= mOutputStrides[k];
                }
                shape_t offset = 0;
                for (shape_t index = 0; index < indexes.size(); ++index) {
                    offset += indexes[index] * mStrides[index];
                }
                *outputData++ = *(inputData + offset);
            }
        }
        return MAI_FAILED;
    }
private:
    enum FLAG {INPUT, PERM, OUTPUT = 0};
    std::vector<shape_t> mStrides;
    std::vector<shape_t> mOutputStrides;
};

void registerTranspose() {
    MAI_REGISTER_OP((OpContext{.opType=TRANSPOSE,}), int32, Transpose);
    MAI_REGISTER_OP((OpContext{.opType=TRANSPOSE,}), int64, Transpose);
    MAI_REGISTER_OP((OpContext{.opType=TRANSPOSE,}), float, Transpose);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
