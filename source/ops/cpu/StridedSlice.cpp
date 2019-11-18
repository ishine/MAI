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
class StridedSlice : public Operator {
public:
    StridedSlice() : mParam(NULL), mRunFirst(true) {}

    ~StridedSlice() {
        MAI_DELETE_PTR(mParam);
    }

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    void setParam(Param* param) override {
        mParam = reinterpret_cast<StridedSliceParam*>(param);
    }

    Param* getParam() override {
        return mParam;
    }

    MAI_STATUS run() override {
        const Tensor* input = getInputTensor(INPUT);
        Tensor* output = getOutputTensor(OUTPUT);

        MAI_OP_RUN_FIRST_START
        const Tensor* begin = getInputTensor(BEGIN);
        const Tensor* end = getInputTensor(END);
        const Tensor* strides = getInputTensor(STRIDES);
        MAI_CHECK_NULL(input);
        MAI_CHECK_NULL(begin);
        MAI_CHECK_NULL(end);
        MAI_CHECK_NULL(strides);
        MAI_CHECK_NULL(output);
        MAI_CHECK_NULL(mParam);

        const int32* beginData = begin->data<int32>();
        const int32* endData = end->data<int32>();
        const int32* stridesData = strides->data<int32>();

        std::vector<int32> beginIndicesV(input->dimSize(), 0);
        std::vector<int32> endIndicesV(input->dimSize(), 0);
        mStridesV.resize(input->dimSize());

        for (int32 i = 0; i < input->dimSize(); ++i) {
            if (i < begin->elementSize()) {
                beginIndicesV[i] = beginData[i];
                endIndicesV[i] = endData[i];
                mStridesV[i] = stridesData[i];
            } else {
                beginIndicesV[i] = 0;
                endIndicesV[i] = input->dim(i);
                mStridesV[i] = 1;
            }
            // TODO: may need workround for begin/end indices if they are out of input dims
            MAI_CHECK(beginIndicesV[i] >= -input->dim(i) && beginIndicesV[i] < input->dim(i),
                    "Invalid begin index %d in dim %d", beginIndicesV[i], i);
            MAI_CHECK(endIndicesV[i] >= -input->dim(i) && endIndicesV[i] <= input->dim(i),
                    "Invalid end index %d in dim %d", endIndicesV[i], i);
        }

        //modify indices by mask
        mBeginIndicesV.resize(input->dimSize());
        mEndIndicesV.resize(input->dimSize());
        std::vector<shape_t> outputShape;
        for (shape_t i = 0; i < input->dimSize(); ++i) {
            if (mParam->beginMask & (1 << i)) {
                mBeginIndicesV[i] = mStridesV[i] > 0 ? 0 : input->dim(i) - 1;
            } else {
                mBeginIndicesV[i] = beginIndicesV[i] < 0 ? beginIndicesV[i] + input->dim(i) : beginIndicesV[i];
            }

            if (mParam->endMask & (1 << i)) {
                mEndIndicesV[i] = mStridesV[i] > 0 ? input->dim(i) : -1;
            } else {
                mEndIndicesV[i] = endIndicesV[i] < 0 ? endIndicesV[i] + input->dim(i) : endIndicesV[i];
            }

            // ignore begin and end mask
            if (mParam->shrinkAxisMask & (1 << i)) {
                mBeginIndicesV[i] = beginIndicesV[i] < 0 ? beginIndicesV[i] + input->dim(i) : beginIndicesV[i];
                mEndIndicesV[i] = mBeginIndicesV[i] + 1;
            }

            shape_t outputDim = ((mEndIndicesV[i] - mBeginIndicesV[i]) + mStridesV[i] - 1)/ mStridesV[i];
            MAI_CHECK(outputDim > 0, "Invalid output dim");

            if (!(mParam->shrinkAxisMask & (1 << i))) {
                outputShape.push_back(outputDim);
            } else {
                MAI_CHECK(outputDim == 1, "Invalid shrink");
            }
        }

        if (outputShape.empty()) {// Scalar
            outputShape.push_back(1);
        }
        output->resize(outputShape);
        MAI_OP_RUN_FIRST_END

        const T* inputData = input->data<T>();
        T* outputData = output->mutableData<T>();
        if (input->dimSize() == 1) {
            for (shape_t i = mBeginIndicesV[0]; mStridesV[0] > 0 ? i < mEndIndicesV[0]
                    : i > mEndIndicesV[0]; i += mStridesV[0]) {
                *outputData++ = inputData[i];
            }
        } else if (input->dimSize() == 2) {
            for (shape_t i = mBeginIndicesV[0]; mStridesV[0] > 0 ? i < mEndIndicesV[0]
                    : i > mEndIndicesV[0]; i += mStridesV[0]) {
                for (shape_t j = mBeginIndicesV[1]; mStridesV[1] > 0 ? j < mEndIndicesV[1]
                        : j > mEndIndicesV[1]; j += mStridesV[1]) {
                    *outputData++ = inputData[i * input->dim(1) + j];
                }
            }
        } else if (input->dimSize() == 3) {
            for (shape_t i = mBeginIndicesV[0]; mStridesV[0] > 0 ? i < mEndIndicesV[0]
                    : i > mEndIndicesV[0]; i += mStridesV[0]) {
                for (shape_t j = mBeginIndicesV[1]; mStridesV[1] > 0 ? j < mEndIndicesV[1]
                        : j > mEndIndicesV[1]; j += mStridesV[1]) {
                    for (shape_t k = mBeginIndicesV[2]; mStridesV[2] > 0 ? k < mEndIndicesV[2]
                            : k > mEndIndicesV[2]; k += mStridesV[2]) {
                        *outputData++ = inputData[(i * input->dim(1) + j) * input->dim(2) + k];
                    }
                }
            }
        } else if (input->dimSize() == 4) {
            for (shape_t i = mBeginIndicesV[0]; mStridesV[0] > 0 ? i < mEndIndicesV[0]
                    : i > mEndIndicesV[0]; i += mStridesV[0]) {
                for (shape_t j = mBeginIndicesV[1]; mStridesV[1] > 0 ? j < mEndIndicesV[1]
                        : j > mEndIndicesV[1]; j += mStridesV[1]) {
                    for (shape_t k = mBeginIndicesV[2]; mStridesV[2] > 0 ? k < mEndIndicesV[2]
                            : k > mEndIndicesV[2]; k += mStridesV[2]) {
                        for (shape_t l = mBeginIndicesV[3]; mStridesV[3] > 0 ? l < mEndIndicesV[3]
                                : l > mEndIndicesV[3]; l += mStridesV[3]) {
                            *outputData++ = inputData[((i * input->dim(1) + j) * input->dim(2) + k) * input->dim(3) + l];
                        }
                    }
                }
            }
        } else {
            MAI_ABORT("Unsupported dim size %d", input->dimSize());
        }

        return MAI_SUCCESS;
    }
private:
    enum FLAG {INPUT, BEGIN, END, STRIDES, OUTPUT = 0};
    StridedSliceParam* mParam;
    bool mRunFirst;
    std::vector<int32> mStridesV;
    std::vector<int32> mBeginIndicesV;
    std::vector<int32> mEndIndicesV;
};

void registerStridedSlice() {
    MAI_REGISTER_OP((OpContextBuilder().setOperatorType(STRIDED_SLICE).build()), float, StridedSlice);
    MAI_REGISTER_OP((OpContextBuilder().setOperatorType(STRIDED_SLICE).build()), int32, StridedSlice);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
