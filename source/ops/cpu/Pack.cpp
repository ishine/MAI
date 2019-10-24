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

class Pack : public Operator {
public:
    Pack() : mNum(0), mAxis(0), mOuterSize(1), mInnerSize(1), mParam(NULL), mRunFirst(true) {}
    ~Pack() {
        MAI_DELETE_PTR(mParam);
    }

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    void setParam(Param* param) override {
        mParam = reinterpret_cast<PackParam*>(param);
        if (mParam) {
            mNum = mParam->num;
            mAxis = mParam->axis;
        }
    }

    Param* getParam() override {
        return mParam;
    }


    MAI_STATUS run() override {
        Tensor* output = getOutputTensor(0);
        MAI_OP_RUN_FIRST_START
        const Tensor* input0 = getInputTensor(0);
        MAI_CHECK((mNum != 0 && inputNames().size() == mNum),
            "Input size(%d) or mNum(%d) is not equal", inputNames().size(), mNum);
        MAI_CHECK(mAxis >= -(input0->dimSize() + 1) && mAxis < input0->dimSize() + 1,
                "Invalid axis:%d should be [%d, %d)", mAxis, -(input0->dimSize() + 1), input0->dimSize() + 1);
        if (mAxis < 0) {
            mAxis += input0->dimSize() + 1;
        }
        MAI_CHECK(input0 != NULL, "input0 is NULL");
        std::vector<shape_t> outputShape(input0->dimSize() + 1);
        for(int32 i = 1; i < mNum; ++i) {
            const Tensor* tmpTensor = getInputTensor(i);
            MAI_CHECK(tmpTensor != NULL, "Input size must be %d, but the %dth is null", mNum, i);
            MAI_CHECK(isShapeSame(input0->shape(), tmpTensor->shape()),
                    "Tensors must have same shape, input0:%s, input%d:%s",
                    shapeToString(input0).c_str(), shapeToString(tmpTensor).c_str());
        }
        int32 index = 0;
        for (shape_t i = 0; i < outputShape.size(); ++i) {
            if (i == mAxis) {
                outputShape[mAxis] = mNum;
            } else {
                outputShape[i] = input0->dim(index++);
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

        char* outputData = output->mutableData<char>();
        int32 sizeofElement = getInputTensor(0)->size() / getInputTensor(0)->elementSize();
        shape_t outputOffset = 0;
        for (shape_t o = 0; o < mOuterSize; ++o) {
            for (int32 i = 0; i < mNum; ++i) {
                const Tensor* tmpTensor = getInputTensor(i);
                MAI_CHECK_NULL(tmpTensor);
                shape_t copySize = mInnerSize * sizeofElement;
                memcpy(outputData + outputOffset, tmpTensor->data<char>() + o * copySize, copySize);
                outputOffset += copySize;
            }
        }
        const int32* dd = output->data<int32>();
        return MAI_SUCCESS;
    }
private:
    int32 mNum;
    int32 mAxis;
    shape_t mOuterSize;
    shape_t mInnerSize;
    PackParam* mParam;
    bool mRunFirst;
};

void registerPack() {
    MAI_REGISTER_OP((OpContext{.opType=PACK,}), Pack);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
