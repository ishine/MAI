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

#pragma once

#include "include/Type.h"

namespace MAI {
namespace Op {
namespace CPU {
namespace Ref {

template<typename T, DataFormat INPUT_FORMAT, DataFormat FILTER_FORMAT>
struct Conv2D {
    static void conv2d(const T* input,
            const std::vector<shape_t>& inputShape,
            const T* filter,
            const std::vector<shape_t>& filterShape,
            const T* bias,
            const std::vector<shape_t>& biasShape,
            const Conv2DParam* param,
            T* output,
            const std::vector<shape_t>& outputShape);
};

template<typename T>
struct Conv2D<T, NCHW, OIHW> {
    static void conv2d(const T* input,
            const std::vector<shape_t>& inputShape,
            const T* filter,
            const std::vector<shape_t>& filterShape,
            const T* bias,
            const std::vector<shape_t>& biasShape,
            const Conv2DParam* param,
            T* output,
            const std::vector<shape_t>& outputShape) {
        int32 outputGroupChannelSize = outputShape[1] / param->group;
        int32 inputGroupChannelSize = inputShape[1] / param->group;
        #pragma omp parallel for collapse(2)
        for(shape_t n = 0; n < outputShape[0]; ++n) {
            for(shape_t o = 0; o < outputShape[1]; ++o) {
                for(shape_t h = 0; h < outputShape[2]; ++h) {
                    for(shape_t w = 0; w < outputShape[3]; ++w) {
                        T* outputV = output + offset4D(outputShape, n, o, h, w);
                        shape_t inHBase = h * param->strides[DataFormatIndex<NCHW>::H] - param->paddings[0];
                        shape_t inWBase = w * param->strides[DataFormatIndex<NCHW>::W] - param->paddings[2];
                        int32 group = o / outputGroupChannelSize;// The group th of output channel
                        for(shape_t i = group * inputGroupChannelSize; i < (group + 1) * inputGroupChannelSize; ++i) {
                            for(shape_t fh = 0; fh < filterShape[DataFormatIndex<OIHW>::H]; ++fh) {
                                for(shape_t fw = 0; fw < filterShape[DataFormatIndex<OIHW>::W]; ++fw) {
                                    shape_t inHOffset = inHBase + fh;
                                    shape_t inWOffset = inWBase + fw;
                                    if (inHOffset >= 0 && inHOffset < inputShape[DataFormatIndex<NCHW>::H]
                                            && inWOffset >= 0 && inWOffset < inputShape[DataFormatIndex<NCHW>::W]) {
                                        shape_t inputOffset = offset4D(inputShape, n, i, inHOffset, inWOffset);
                                        shape_t filterOffset = offset4D(filterShape, o, i % inputGroupChannelSize, fh, fw);
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
};

template<typename T>
struct Conv2D<T, NHWC, HWIO> {
    static void conv2d(const T* input,
            const std::vector<shape_t>& inputShape,
            const T* filter,
            const std::vector<shape_t>& filterShape,
            const T* bias,
            const std::vector<shape_t>& biasShape,
            const Conv2DParam* param,
            T* output,
            const std::vector<shape_t>& outputShape) {
        int32 outputGroupChannelSize = outputShape[DataFormatIndex<NHWC>::C] / param->group;
        int32 inputGroupChannelSize = inputShape[DataFormatIndex<NHWC>::C] / param->group;
        #pragma omp parallel for collapse(4)
        for(shape_t n = 0; n < outputShape[0]; ++n) {
            for(shape_t h = 0; h < outputShape[1]; ++h) {
                for(shape_t w = 0; w < outputShape[2]; ++w) {
                    for(shape_t o = 0; o < outputShape[3]; ++o) {
                        T* outputV = output + offset4D(outputShape, n, h, w, o);
                        shape_t inHBase = h * param->strides[DataFormatIndex<NHWC>::H] - param->paddings[0];
                        shape_t inWBase = w * param->strides[DataFormatIndex<NHWC>::W] - param->paddings[2];
                        int32 group = o / outputGroupChannelSize;// The group th of output channel
                        for(shape_t i = group * inputGroupChannelSize; i < (group + 1) * inputGroupChannelSize; ++i) {
                            for(shape_t fh = 0; fh < filterShape[DataFormatIndex<HWIO>::H]; ++fh) {
                                for(shape_t fw = 0; fw < filterShape[DataFormatIndex<HWIO>::H]; ++fw) {
                                    shape_t inHOffset = inHBase + fh;
                                    shape_t inWOffset = inWBase + fw;
                                    if (inHOffset >= 0 && inHOffset < inputShape[DataFormatIndex<NHWC>::H]
                                            && inWOffset >= 0 && inWOffset < inputShape[DataFormatIndex<NHWC>::W]) {
                                        shape_t inputOffset = offset4D(inputShape, n, inHOffset, inWOffset, i);
                                        shape_t filterOffset = offset4D(filterShape, fh, fw, i % inputGroupChannelSize, o);
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
};
} // namespace Ref
} // namespace CPU
} // namespace Op
} // namespace MAI
