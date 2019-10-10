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
    Pool(MAIOperator poolType, PoolFunction fNHWC, PoolFunction fNCHW)
        : mPoolType(poolType), mFunctionNHWC(fNHWC), mFunctionNCHW(fNCHW), mParam(NULL), mRunFirst(true) {
    }
    ~Pool() {
        if (mParam) {
            delete mParam;
            mParam = NULL;
        }
    }

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    void setParam(Param* param) override {
        mParam = reinterpret_cast<PoolParam*>(param);
    }

    MAI_STATUS run() override {
        MAI_OP_RUN_FIRST_START
        mInput = getInputTensor(INPUT);
        mOutput = getOutputTensor(OUTPUT);
        MAI_CHECK_NULL(mInput);
        MAI_CHECK_NULL(mOutput);
        if (mPoolType == GLOBAL_AVG_POOL) {
            if (mParam != NULL) {
                delete mParam;
            }
            mParam = new PoolParam();
            mParam->paddingMode = VALID;
            mParam->strides = {1,1,1,1};
        }
        MAI_CHECK_NULL(mParam);
        MAI_CHECK(mInput->shape().size() == 4, "Input shape must be 4-d");
        if (mParam->paddingMode != INVALID) {
            MAI_CHECK(mParam->paddings.size() == 0,
                "Cannot use explicit padding when paddingMode is :%d", mParam->paddingMode);
        } else {
            MAI_CHECK(mParam->paddings.size() == 4,
                "Explicit padding size must be 4 but not: %d", mParam->paddings.size());
        }

        std::vector<shape_t> outputShape(4);
        if (mInput->getDataFormat() == NHWC) {
            if (mPoolType == GLOBAL_AVG_POOL) {
                mParam->kernelSizes = {1, mInput->dimH(), mInput->dimW(), 1};
            }
            MAI_CHECK(mParam->kernelSizes.size() == 4, "KernelSize must be 4-d format");
            mFunction = mFunctionNHWC;
        } else if (mInput->getDataFormat() == NCHW) {
            if (mPoolType == GLOBAL_AVG_POOL) {
                mParam->kernelSizes = {1, 1, mInput->dimH(), mInput->dimW()};
            }
            MAI_CHECK(mParam->kernelSizes.size() == 4, "KernelSize must be 4-d format");
            mFunction = mFunctionNCHW;

        } else {
            MAI_ABORT("Unsupported dataFormat:%s", getNameFromDataFormat(mInput->getDataFormat()));
        }
        outputShape[mInput->n()] = mInput->dimN();
        outputShape[mInput->c()] = mInput->dimC();
        std::vector<int32> outputHW = calculateHW(
                {mInput->dimH(), mInput->dimW()},
                {mParam->kernelSizes[mInput->h()], mParam->kernelSizes[mInput->w()]},
                {mParam->strides[mInput->h()], mParam->strides[mInput->w()]},
                mParam->paddings, mParam->paddingMode);
        outputShape[mInput->h()] = outputHW[0];
        outputShape[mInput->w()] = outputHW[1];
        mOutput->resize(outputShape);
        if (mParam->paddingMode != INVALID) {
            mParam->paddings = calcPaddings(mParam->paddingMode, mParam->kernelSizes);
        }

        if (mFunction == NULL) {
            MAI_CHECK(false, "Unsupported input data format: %s", getNameFromDataFormat(mInput->getDataFormat()).c_str());
        }

        MAI_OP_RUN_FIRST_END

        mOutput->zero();
        mFunction(mInput->data<T>(), mInput->shape(),
                mParam,
                mOutput->mutableData<T>(), mOutput->shape());
        return MAI_SUCCESS;
    }

private:
    enum FLAG {INPUT, OUTPUT = 0,};
    const Tensor* mInput;
    Tensor* mOutput;
    MAIOperator mPoolType;
    PoolFunction mFunction;
    PoolFunction mFunctionNHWC;
    PoolFunction mFunctionNCHW;
    PoolParam* mParam;
    bool mRunFirst;
};

template<typename T>
class MaxPool : public Pool<T> {
public:
    MaxPool() : Pool<T>(MAX_POOL, poolNHWC, poolNCHW){
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
                        for (shape_t fh = 0; fh < param->kernelSizes[DataFormatIndex<NHWC>::H]; ++fh) {
                            for (shape_t fw = 0; fw < param->kernelSizes[DataFormatIndex<NHWC>::W]; ++fw) {
                                shape_t iHOffset = iHBase + fh;
                                shape_t iWOffset = iWBase + fw;
                                if (iHOffset >= 0 && iHOffset < inputShape[DataFormatIndex<NHWC>::H]
                                        && iWOffset >= 0 && iWOffset < inputShape[DataFormatIndex<NHWC>::W]) {
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
        for(shape_t n = 0; n < outputShape[0]; ++n) {
            for(shape_t c = 0; c < outputShape[1]; ++c) {
                for(shape_t h = 0; h < outputShape[2]; ++h) {
                    for(shape_t w = 0; w < outputShape[3]; ++w) {
                        shape_t iHBase = h * param->strides[DataFormatIndex<NCHW>::H] - param->paddings[0];
                        shape_t iWBase = w * param->strides[DataFormatIndex<NCHW>::W] - param->paddings[2];
                        T max = std::numeric_limits<T>::lowest();
                        int count = 0;
                        for (shape_t fh = 0; fh < param->kernelSizes[DataFormatIndex<NCHW>::H]; ++fh) {
                            for (shape_t fw = 0; fw < param->kernelSizes[DataFormatIndex<NCHW>::W]; ++fw) {
                                shape_t iHOffset = iHBase + fh;
                                shape_t iWOffset = iWBase + fw;
                                if (iHOffset >= 0 && iHOffset < inputShape[DataFormatIndex<NCHW>::H]
                                        && iWOffset >= 0 && iWOffset < inputShape[DataFormatIndex<NCHW>::W]) {
                                    const T* inputV = input + offset4D(inputShape, n, c, iHOffset, iWOffset);
                                    if (*inputV > max) {
                                        max = *inputV;
                                    }
                                    count++;
                                }
                            }
                        }
                        T* outputV = output + offset4D(outputShape, n, c, h, w);
                        *outputV = count == 0 ? 0 : max;
                    }
                }
            }
        }
    }

};

template<typename T>
class AvgPool : public Pool<T> {
public:
    AvgPool() : Pool<T>(AVG_POOL, poolNHWC, poolNCHW){
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
                        for (shape_t fh = 0; fh < param->kernelSizes[DataFormatIndex<NHWC>::H]; ++fh) {
                            for (shape_t fw = 0; fw < param->kernelSizes[DataFormatIndex<NHWC>::W]; ++fw) {
                                shape_t iHOffset = iHBase + fh;
                                shape_t iWOffset = iWBase + fw;
                                if (iHOffset >= 0 && iHOffset < inputShape[DataFormatIndex<NHWC>::H]
                                        && iWOffset >= 0 && iWOffset < inputShape[DataFormatIndex<NHWC>::W]) {
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
        for(shape_t n = 0; n < outputShape[0]; ++n) {
            for(shape_t c = 0; c < outputShape[1]; ++c) {
                for(shape_t h = 0; h < outputShape[2]; ++h) {
                    for(shape_t w = 0; w < outputShape[3]; ++w) {
                        shape_t iHBase = h * param->strides[DataFormatIndex<NCHW>::H] - param->paddings[0];
                        shape_t iWBase = w * param->strides[DataFormatIndex<NCHW>::W] - param->paddings[2];
                        T sum = 0;
                        int32 count = 0;
                        for (shape_t fh = 0; fh < param->kernelSizes[DataFormatIndex<NCHW>::H]; ++fh) {
                            for (shape_t fw = 0; fw < param->kernelSizes[DataFormatIndex<NCHW>::W]; ++fw) {
                                shape_t iHOffset = iHBase + fh;
                                shape_t iWOffset = iWBase + fw;
                                if (iHOffset >= 0 && iHOffset < inputShape[DataFormatIndex<NCHW>::H]
                                        && iWOffset >= 0 && iWOffset < inputShape[DataFormatIndex<NCHW>::W]) {
                                    const T* inputV = input + offset4D(inputShape, n, c, iHOffset, iWOffset);
                                    sum += *inputV;
                                    count++;
                                }
                            }
                        }
                        T* outputV = output + offset4D(outputShape, n, c, h, w);
                        *outputV = count == 0 ? 0 : (sum / count);
                    }
                }
            }
        }
    }

};

template<class T>
class GlobalAvgPool : public Pool<T> {
public:
    GlobalAvgPool() : Pool<T>(GLOBAL_AVG_POOL, AvgPool<T>::poolNHWC, AvgPool<T>::poolNCHW){
    }
};

void registerMaxPool() {
    MAI_REGISTER_OP((OpContext{.opType=MAX_POOL,}), float, MaxPool);
}

void registerAvgPool() {
    MAI_REGISTER_OP((OpContext{.opType=AVG_POOL,}), float, AvgPool);
}

void registerGlobalAvgPool() {
    MAI_REGISTER_OP((OpContext{.opType=GLOBAL_AVG_POOL,}), float, GlobalAvgPool);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
