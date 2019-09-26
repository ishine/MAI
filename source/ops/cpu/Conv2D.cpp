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
class Conv2D : public Operator {
public:
    Conv2D() = default;
    ~Conv2D() {
        delete mParam;
    }

    MAI_STATUS init() override {
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
                "Cannot use explicit padding when paddingMode is :%d", mParam->paddingMode);
        } else {
            MAI_CHECK(mParam->paddings.size() == 4,
                "Explicit padding size must be 4 but not: %d", mParam->paddings.size());
        }

        std::vector<shape_t> outputShape(4);
        if (mInput->getDataFormat() == NHWC) {
            if (mFilter->getDataFormat() == HWIO) {
                outputShape[0] = mInput->dim(0);
                outputShape[3] = mFilter->dim(3);
                std::vector<int32> outputHW = calculateHW(
                        {mInput->dim(DataFormatIndex<NHWC>::H), mInput->dim(DataFormatIndex<NHWC>::W)},
                        {mFilter->dim(DataFormatIndex<HWIO>::H), mFilter->dim(DataFormatIndex<HWIO>::W)},
                        mParam->strides, mParam->paddings, mParam->paddingMode);
                outputShape[1] = outputHW[0];
                outputShape[2] = outputHW[1];
                mFunction = conv2DNHWC_HWIO;
            }
            if (mParam->paddingMode != INVALID) {
                mParam->paddings = calcPaddings(mParam->paddingMode,
                        {mFilter->dim(DataFormatIndex<HWIO>::H), mFilter->dim(DataFormatIndex<HWIO>::W)});
            }
        }
        mOutput->resize(outputShape);
        mOutput->zero();

        if (mFunction == NULL) {
            MAI_CHECK(false, "Unsupported input data format: %s, with filter data format:%s",
                    getNameFromDataFormat(mInput->getDataFormat()).c_str(),
                    getNameFromDataFormat(mFilter->getDataFormat()).c_str());
        }

        return MAI_SUCCESS;
    }

    void setParam(Param* param) override {
        mParam = reinterpret_cast<Conv2DParam*>(param);
    }

    static void conv2DNHWC_HWIO(const T* input,
            const std::vector<shape_t>& inputShape,
            const T* filter,
            const std::vector<shape_t>& filterShape,
            const T* bias,
            const std::vector<shape_t>& biasShape,
            const Conv2DParam* param,
            T* output,
            const std::vector<shape_t>& outputShape) {
        for(shape_t n = 0; n < outputShape[0]; ++n) {
            for(shape_t h = 0; h < outputShape[1]; ++h) {
                for(shape_t w = 0; w < outputShape[2]; ++w) {
                    for(shape_t o = 0; o < outputShape[3]; ++o) {
                        T* outputV = output + offset4D(outputShape, n, h, w, o);
                        shape_t inHBase = h * param->strides[1] - param->paddings[0];
                        shape_t inWBase = w * param->strides[2] - param->paddings[2];
                        for(shape_t i = 0; i < inputShape[3]; ++i) {
                            for(shape_t fh = 0; fh < filterShape[0]; ++fh) {
                                for(shape_t fw = 0; fw < filterShape[1]; ++fw) {
                                    shape_t inHOffset = inHBase + fh;
                                    shape_t inWOffset = inWBase + fw;
                                    if (inHOffset >= 0 && inHOffset < inputShape[1]
                                            && inWOffset >= 0 && inWOffset < inputShape[2]) {
                                        shape_t inputOffset = offset4D(inputShape, n, inHOffset, inWOffset, i);
                                        shape_t filterOffset = offset4D(filterShape, fh, fw, i, o);
                                        const T* inputV = input + inputOffset;
                                        const T* filterV = filter + filterOffset;
                                        *outputV += (*inputV) * (*filterV);
                                    }
                                }
                            }
                        }
                        if (bias != NULL) {
                            *outputV += *(bias + o);
                        }
                    }
                }
            }
        }
    }

    static void conv2DNCHW_OIHW(const T* input,
            const std::vector<shape_t>& inputShape,
            const T* filter,
            const std::vector<shape_t>& filterShape,
            const T* bias,
            const std::vector<shape_t>& biasShape,
            const Conv2DParam* param,
            T* output,
            const std::vector<shape_t>& outputShape) {
        for(shape_t n = 0; n < outputShape[0]; ++n) {
            for(shape_t h = 0; h < outputShape[1]; ++h) {
                for(shape_t w = 0; w < outputShape[2]; ++w) {
                    for(shape_t o = 0; o < outputShape[3]; ++o) {
                        T* outputV = output + offset4D(outputShape, n, h, w, o);
                        shape_t inHBase = h * param->strides[1] - param->paddings[0];
                        shape_t inWBase = w * param->strides[2] - param->paddings[2];
                        for(shape_t i = 0; i < inputShape[3]; ++i) {
                            for(shape_t fh = 0; fh < filterShape[0]; ++fh) {
                                for(shape_t fw = 0; fw < filterShape[1]; ++fw) {
                                    shape_t inHOffset = inHBase + fh;
                                    shape_t inWOffset = inWBase + fw;
                                    if (inHOffset >= 0 && inHOffset < inputShape[1]
                                            && inWOffset >= 0 && inWOffset < inputShape[2]) {
                                        shape_t inputOffset = offset4D(inputShape, n, inHOffset, inWOffset, i);
                                        shape_t filterOffset = offset4D(filterShape, fh, fw, i, o);
                                        const T* inputV = input + inputOffset;
                                        const T* filterV = filter + filterOffset;
                                        *outputV += (*inputV) * (*filterV);
                                    }
                                }
                            }
                        }
                        if (bias != NULL) {
                            *outputV += *(bias + o);
                        }
                    }
                }
            }
        }
    }

    MAI_STATUS run() override {
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
};

void registerConv2D() {
    MAI_REGISTER_OP((OpContext{.opType=CONV2D,}), float, Conv2D);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
