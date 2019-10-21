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

#include <algorithm>
#include "core/OperatorRegister.h"
#include "util/MAIUtil.h"
//#ifdef MAI_NEON_ENABLED
//#include "neon/TransposeNeon.h"
//#else
#include "ref/Conv2DRef.h"
//#endif

namespace MAI {
namespace Op {
namespace CPU {

template<typename T>
class Conv2D : public Operator {
public:
    Conv2D() : mParam(NULL), mRunFirst(true) {}
    ~Conv2D() {
        if (mParam != NULL) {
            delete mParam;
            mParam = NULL;
        }
    }

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    void setParam(Param* param) override {
        mParam = reinterpret_cast<Conv2DParam*>(param);
    }

    MAI_STATUS run() override {
        MAI_OP_RUN_FIRST_START
        mInput = getInputTensor(INPUT);
        mFilter = getInputTensor(FILTER);
        mBias = getInputTensor(BIAS);
        mOutput = getOutputTensor(OUTPUT);
        MAI_CHECK_NULL(mInput);
        MAI_CHECK_NULL(mFilter);
        MAI_CHECK_NULL(mOutput);
        MAI_CHECK_NULL(mParam);
        MAI_CHECK(mInput->shape().size() == 4, "Input shape must be 4-d");
        MAI_CHECK(checkVectorValues(mParam->dilations, 1), "Cannot support dilations greater than 1 now");
        if (mParam->paddingMode != INVALID) {
            MAI_CHECK(mParam->paddings.size() == 0,
                "Cannot use explicit padding when paddingMode is :%d, size:%d", mParam->paddingMode, mParam->paddings.size());
        } else {
            MAI_CHECK(mParam->paddings.size() == 4,
                "Explicit padding size must be 4 but not: %d", mParam->paddings.size());
        }

        std::vector<shape_t> outputShape(4);
        MAI_CHECK(mInput->dimC() == (mFilter->dimI() * mParam->group),
                "input channel(%d) should be equal to filter channel(%d)", mInput->dimC(),
                mFilter->dimI());
        MAI_CHECK(mFilter->dimO() % mParam->group == 0, "filter output channel(%d) must be devided by group(%d)",
                mFilter->dimO(), mParam->group);
        outputShape[mInput->n()] = mInput->dimN();
        outputShape[mInput->c()] = mFilter->dimO();
        std::vector<int32> outputHW = calculateHW(
                {mInput->dimH(), mInput->dimW()},
                {mFilter->dimH(), mFilter->dimW()},
                {mParam->strides[mInput->h()], mParam->strides[mInput->w()]},
                mParam->paddings, mParam->paddingMode);
        outputShape[mInput->h()] = outputHW[0];
        outputShape[mInput->w()] = outputHW[1];
        if (mInput->getDataFormat() == NHWC) {
            if (mFilter->getDataFormat() == HWIO) {
                mFunction = Ref::Conv2D<T, NHWC, HWIO>::conv2d;
            }
            if (mParam->paddingMode != INVALID) {
                mParam->paddings = calcPaddings(mParam->paddingMode,
                        {mFilter->dimH(), mFilter->dimW()});
            }
        } else if (mInput->getDataFormat() == NCHW) {
            if (mFilter->getDataFormat() == OIHW) {
                mFunction = Ref::Conv2D<T, NCHW, OIHW>::conv2d;
            }
        }
        if (mFunction == NULL) {
            MAI_ABORT("Unsupported InputFormat:%s with FilterFormat:%s",
                    getNameFromDataFormat(mInput->getDataFormat()).c_str(),
                    getNameFromDataFormat(mFilter->getDataFormat()).c_str());
        }

        mOutput->resize(outputShape);

        if (mFunction == NULL) {
            MAI_CHECK(false, "Unsupported input data format: %s, with filter data format:%s",
                    getNameFromDataFormat(mInput->getDataFormat()).c_str(),
                    getNameFromDataFormat(mFilter->getDataFormat()).c_str());
        }
        MAI_OP_RUN_FIRST_END

        mOutput->zero();
        std::vector<shape_t> biasShape;
        if (mBias != NULL) {
            biasShape = mBias->shape();
        }
        mFunction(mInput->data<T>(), mInput->shape(),
                mFilter->data<T>(), mFilter->shape(),
                mBias == NULL ? NULL : mBias->data<T>(), biasShape,
                mParam,
                mOutput->mutableData<T>(), mOutput->shape());
        return MAI_SUCCESS;
    }

private:
    enum FLAG {INPUT, FILTER, BIAS, OUTPUT = 0,};
    const Tensor* mInput;
    const Tensor* mFilter;
    const Tensor* mBias;
    Tensor* mOutput;
    std::function<void(const T*, const std::vector<shape_t>&,
            const T*, const std::vector<shape_t>&,
            const T*, const std::vector<shape_t>&,
            const Conv2DParam*,
            T*, const std::vector<shape_t>&)> mFunction;
    Conv2DParam* mParam;
    bool mRunFirst;
};

void registerConv2D() {
    MAI_REGISTER_OP((OpContext{.opType=CONV2D,}), float, Conv2D);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
