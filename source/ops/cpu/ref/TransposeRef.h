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

template<typename T, int32 DIMSIZE>
struct Transpose {
    static void transpose(const std::vector<shape_t>& inputShape, const T* input,
            const int32* perm,
            const std::vector<shape_t>& outputShape, T* output, int32 sizeOfElement = sizeof(T));
};

/**
 * This is a navie function
 */
template<typename T>
struct Transpose<T, 2> {
    static void transpose(const std::vector<shape_t>& inputShape, const T* input,
            const int32* perm,
            const std::vector<shape_t>& outputShape, T* output, int32 sizeOfElement = sizeof(T)) {
        const shape_t H = inputShape[0];
        const shape_t W = inputShape[1];
        for (shape_t h = 0; h < H; ++h) {
            for (shape_t w = 0; w < W; ++w) {
                output[w * H + h] = input[h * H + w];
            }
        }
    }
};

/**
 * This can receive any dim size array which with any type
 */
template<>
struct Transpose<uint8, MAI_DYNAMIC_DIM> {
    static void transpose(const std::vector<shape_t>& inputShape, const uint8* input,
            const int32* perm,
            const std::vector<shape_t>& outputShape, uint8* output, int32 sizeOfElement) {
        std::vector<shape_t> strides;
        std::vector<shape_t> outputStrides;
        std::vector<shape_t> inputStrides(inputShape.size());
        shape_t gap = 1;
        for (int32 i = inputShape.size() - 1; i >= 0; --i) {
            inputStrides[i] = gap;
            gap *= inputShape[i];
        }
        strides.resize(inputShape.size());
        for (shape_t i = 0; i < inputShape.size(); ++i) {
            strides[i] = inputStrides[perm[i]];
        }
        outputStrides.resize(inputShape.size());
        gap = 1;
        for (int32 i = outputShape.size() - 1; i >= 0; --i) {
            outputStrides[i] = gap;
            gap *= outputShape[i];
        }
        shape_t loopCount = 1;
        std::vector<shape_t> indexes(outputShape.size(), 0);
        for (int32 i = outputShape.size() - 1; i >= 0; --i) {
            for (shape_t k = 0; k < i; ++k) {
                indexes[k] = 0;
            }
            shape_t innerSize = 1;
            if (i == outputShape.size() - 1) {
                innerSize = 0;
            } else {
                for (shape_t k = i + 1; k < outputShape.size(); ++k) {
                    innerSize *= outputShape[k];
                }
            }
            loopCount *= outputShape[i];
            for (shape_t j = innerSize; j < loopCount; ++j) {
                shape_t tmpLoopCount = j;
                for(shape_t k = i; k < outputShape.size(); ++k) {
                    indexes[k] = tmpLoopCount / outputStrides[k];
                    tmpLoopCount %= outputStrides[k];
                }
                shape_t offset = 0;
                for (shape_t index = 0; index < indexes.size(); ++index) {
                    offset += indexes[index] * strides[index];
                }
                memcpy(output, input + offset * sizeOfElement, sizeOfElement);
                output += sizeOfElement;
            }
        }
    }
};

} // namespace Ref
} // namespace CPU
} // namespace Op
} // namespace MAI
