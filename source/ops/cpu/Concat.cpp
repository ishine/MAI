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

class Concat : public Operator {
public:
    Concat() : mNum(0), mAxis(0), mOuterSize(1), mInnerSize(1), mRunFirst(true) {}
    ~Concat() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    void setParam(Param* param) override {
        ConcatParam* concatParam = reinterpret_cast<ConcatParam*>(param);
        if (concatParam) {
            mNum = concatParam->num;
            mAxis = concatParam->axis;
        }
        delete param;
    }


    MAI_STATUS run() override {
        Tensor* output = getOutputTensor(0);
        MAI_OP_RUN_FIRST_START
        const Tensor* input0 = getInputTensor(0);
        MAI_CHECK((mNum != 0 && inputNames().size() == mNum),
            "Input size(%d) or mNum(%d) is not equal", inputNames().size(), mNum);
        MAI_CHECK(mAxis >= -(input0->dimSize()) && mAxis < input0->dimSize(), "Invalid axis:%d", mAxis);
        if (mAxis < 0) {
            mAxis += input0->dimSize();
        }
        MAI_CHECK(input0 != NULL, "input0 is NULL");
        std::vector<shape_t> outputShape(input0->shape());
        for(int32 i = 1; i < mNum; ++i) {
            const Tensor* tmpTensor = getInputTensor(i);
            MAI_CHECK(tmpTensor != NULL, "Input size must be %d, but the %dth is null", mNum, i);
            MAI_CHECK(outputShape.size() == tmpTensor->dimSize(), "rank must be equal, while input0 is %d, input%d, is %d",
                    outputShape.size(), i, tmpTensor->dimSize());
            for (shape_t i = 0; i < outputShape.size(); ++i) {
                if (i == mAxis) {
                    outputShape[mAxis] += tmpTensor->dim(i);
                } else {
                    MAI_CHECK(outputShape[i] == tmpTensor->dim(i), "dim must equal(Expect %d real is %d)", outputShape[i], tmpTensor->dim(i));
                }
            }
        }
        MAI_CHECK_NULL(output);
        output->resize(outputShape);

        for (shape_t i = 0; i < outputShape.size(); ++i) {
            if (i < (shape_t)mAxis) {
                mOuterSize *= outputShape[i];
            } else if (i > (shape_t)mAxis) {
                mInnerSize *= outputShape[i];
            }
        }
        MAI_OP_RUN_FIRST_END

        void* outputData = output->mutableData<void>();
        int32 sizeofElement = getInputTensor(0)->size() / getInputTensor(0)->elementSize();
        shape_t outputOffset = 0;
        for (shape_t o = 0; o < mOuterSize; ++o) {
            for (int32 i = 0; i < mNum; ++i) {
                const Tensor* tmpTensor = getInputTensor(i);
                MAI_CHECK_NULL(tmpTensor);
                shape_t copySize = tmpTensor->dim(mAxis) * mInnerSize * sizeofElement;
                memcpy(outputData + outputOffset, tmpTensor->data<void>() + o * copySize, copySize);
                outputOffset += copySize;
            }
        }
        return MAI_SUCCESS;
    }
private:
    int32 mNum;
    int32 mAxis;
    shape_t mOuterSize;
    shape_t mInnerSize;
    bool mRunFirst;
};

void registerConcat() {
    MAI_REGISTER_OP((OpContext{.opType=CONCAT,}), Concat);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
