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

namespace MAI {
namespace Op {
namespace CPU {

template<typename T>
class Pool : public Operator {
public:
    typedef std::function<void(const T*, const std::vector<shape_t>&,
            const PoolParam*,
            T*, const std::vector<shape_t>&)> PoolFunction;
    Pool(PoolFunction fNHWC, PoolFunction fNCHW) : mFunctionNHWC(fNHWC), mFunctionNCHW(fNCHW) {
    }
    ~Pool() {
        delete mParam;
    }

    MAI_STATUS init() override {
        mInput = getInputTensor(INPUT);
        mOutput = getOutputTensor(OUTPUT);
        MAI_CHECK_NULL(mInput);
        MAI_CHECK_NULL(mOutput);
        MAI_CHECK_NULL(mParam);
        MAI_CHECK(mInput->shape().size() == 4, "Input shape must be 4-d");
        MAI_CHECK(mParam->kernelSizes.size() == 4, "KernelSize must be 4-d format");
        if (mParam->paddingMode != INVALID) {
            MAI_CHECK(mParam->paddings.size() == 0,
                "Cannot use explicit padding when paddingMode is :%d", mParam->paddingMode);
        } else {
            MAI_CHECK(mParam->paddings.size() == 4,
                "Explicit padding size must be 4 but not: %d", mParam->paddings.size());
        }

        std::vector<shape_t> outputShape(4);
        if (mInput->getDataFormat() == NHWC) {
            outputShape[0] = mInput->dim(0);
            outputShape[3] = mInput->dim(3);
            std::vector<int32> outputHW = calculateHW(
                    {mInput->dim(1), mInput->dim(2)},
                    {mParam->kernelSizes[1], mParam->kernelSizes[2]},
                    mParam->strides, mParam->paddings, mParam->paddingMode);
            outputShape[1] = outputHW[0];
            outputShape[2] = outputHW[1];
            mFunction = mFunctionNHWC;
            if (mParam->paddingMode != INVALID) {
                mParam->paddings = calcPaddings(mParam->paddingMode, mParam->kernelSizes);
            }
        }
        mOutput->resize(outputShape);
        mOutput->zero();

        if (mFunction == NULL) {
            MAI_CHECK(false, "Unsupported input data format: %d", mInput->getDataFormat());
        }

        return MAI_SUCCESS;
    }

    void setParam(Param* param) override {
        mParam = reinterpret_cast<PoolParam*>(param);
    }

    MAI_STATUS run() override {
        mFunction(mInput->data<T>(), mInput->shape(),
                mParam,
                mOutput->mutableData<T>(), mOutput->shape());
        return MAI_SUCCESS;
    }

private:
    enum FLAG {INPUT, OUTPUT = 0,};
    const Tensor* mInput;
    Tensor* mOutput;
    PoolFunction mFunction;
    PoolFunction mFunctionNHWC;
    PoolFunction mFunctionNCHW;
    PoolParam* mParam;
};

template<typename T>
class MaxPool : public Pool<T> {
public:
    MaxPool() : Pool<T>(poolNHWC, poolNCHW){
    }

    static void poolNHWC(const T* input,
            const std::vector<shape_t>& inputShape,
            const PoolParam* param,
            T* output,
            const std::vector<shape_t>& outputShape) {
        for(shape_t n = 0; n < outputShape[0]; ++n) {
            for(shape_t h = 0; h < outputShape[1]; ++h) {
                for(shape_t w = 0; w < outputShape[2]; ++w) {
                    for(shape_t c = 0; c < outputShape[3]; ++c) {
                        shape_t iHBase = h * param->strides[DataFormatIndex<NHWC>::H] - param->paddings[0];
                        shape_t iWBase = w * param->strides[DataFormatIndex<NHWC>::W] - param->paddings[2];
                        T max = std::numeric_limits<T>::lowest();
                        int count = 0;
                        for (shape_t fh = 0; fh < param->kernelSizes[1]; ++fh) {
                            for (shape_t fw = 0; fw < param->kernelSizes[1]; ++fw) {
                                shape_t iHOffset = iHBase + fh;
                                shape_t iWOffset = iWBase + fw;
                                if (iHOffset >= 0 && iHOffset < inputShape[1]
                                        && iWOffset >= 0 && iWOffset < inputShape[2]) {
                                    const T* inputV = input + offset4D(inputShape, n, iHOffset, iWOffset, c);
                                    if (*inputV > max) {
                                        max = *inputV;
                                    }
                                    count++;
                                }
                            }
                        }
                        T* outputV = output + offset4D(outputShape, n, h, w, c);
                        *outputV = count == 0 ? 0 : max;
                    }
                }
            }
        }
    }

    static void poolNCHW(const T* input,
            const std::vector<shape_t>& inputShape,
            const PoolParam* param,
            T* output,
            const std::vector<shape_t>& outputShape) {
    }

};

template<typename T>
class AvgPool : public Pool<T> {
public:
    AvgPool() : Pool<T>(poolNHWC, poolNCHW){
    }
    static void poolNHWC(const T* input,
            const std::vector<shape_t>& inputShape,
            const PoolParam* param,
            T* output,
            const std::vector<shape_t>& outputShape) {
        for(shape_t n = 0; n < outputShape[0]; ++n) {
            for(shape_t h = 0; h < outputShape[1]; ++h) {
                for(shape_t w = 0; w < outputShape[2]; ++w) {
                    for(shape_t c = 0; c < outputShape[3]; ++c) {
                        shape_t iHBase = h * param->strides[DataFormatIndex<NHWC>::H] - param->paddings[0];
                        shape_t iWBase = w * param->strides[DataFormatIndex<NHWC>::W] - param->paddings[2];
                        T sum = 0;
                        int32 count = 0;
                        for (shape_t fh = 0; fh < param->kernelSizes[1]; ++fh) {
                            for (shape_t fw = 0; fw < param->kernelSizes[1]; ++fw) {
                                shape_t iHOffset = iHBase + fh;
                                shape_t iWOffset = iWBase + fw;
                                if (iHOffset >= 0 && iHOffset < inputShape[1]
                                        && iWOffset >= 0 && iWOffset < inputShape[2]) {
                                    const T* inputV = input + offset4D(inputShape, n, iHOffset, iWOffset, c);
                                    sum += *inputV;
                                    count++;
                                }
                            }
                        }
                        T* outputV = output + offset4D(outputShape, n, h, w, c);
                        *outputV = count == 0 ? 0 : (sum / count);
                    }
                }
            }
        }
    }

    static void poolNCHW(const T* input,
            const std::vector<shape_t>& inputShape,
            const PoolParam* param,
            T* output,
            const std::vector<shape_t>& outputShape) {
    }

};

void registerMaxPool() {
    MAI_REGISTER_OP((OpContext{.opType=MAX_POOL,}), float, MaxPool);
}

void registerAvgPool() {
    MAI_REGISTER_OP((OpContext{.opType=AVG_POOL,}), float, AvgPool);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
