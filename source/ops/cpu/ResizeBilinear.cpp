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
class ResizeBilinear : public Operator {
public:
    ResizeBilinear() : mRunFirst(true) {}
    ~ResizeBilinear() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    void setParam(Param* param) override {
    }

    static void resizeNHWC(const std::vector<shape_t>& inputShape, const T* input,
            const std::vector<shape_t>& outputShape, T* output) {
        auto computeLower = [](shape_t inSize, shape_t outSize, shape_t index) -> shape_t {
            float scale = inSize / static_cast<float>(outSize);
            return static_cast<shape_t>(index * scale);
        };
        auto computeUpper = [](shape_t inSize, shape_t outSize, shape_t index) -> shape_t {
            float scale = inSize / static_cast<float>(outSize);
            return static_cast<shape_t>(std::min(static_cast<shape_t>(index * scale) + 1, inSize - 1));
        };
        auto computeDeltaR = [](shape_t inSize, shape_t outSize, shape_t index) -> float {
            float scale = inSize / static_cast<float>(outSize);
            return index * scale - static_cast<shape_t>(index * scale);
        };

        for (shape_t n = 0; n < outputShape[0]; ++n) {
            for (shape_t h = 0; h < outputShape[1]; ++h) {
                for (shape_t w = 0; w < outputShape[2]; ++w) {
                    const T* topLeft = input + offset4D(inputShape, n, computeLower(inputShape[1], outputShape[1], h), computeLower(inputShape[2], outputShape[2], w), 0);
                    const T* topRight = input + offset4D(inputShape, n, computeLower(inputShape[1], outputShape[1], h), computeUpper(inputShape[2], outputShape[2], w), 0);
                    const T* bottomLeft = input + offset4D(inputShape, n, computeUpper(inputShape[1], outputShape[1], h), computeLower(inputShape[2], outputShape[2], w), 0);
                    const T* bottomRight = input + offset4D(inputShape, n, computeUpper(inputShape[1], outputShape[1], h), computeUpper(inputShape[2], outputShape[2], w), 0);
                    float deltaRH = computeDeltaR(inputShape[1], outputShape[1], h);
                    float deltaRW = computeDeltaR(inputShape[2], outputShape[2], w);
                    T* outputData = output + offset4D(outputShape, n, h, w, 0);
                    for (shape_t c = 0; c < outputShape[3]; ++c) {
                        outputData[c] = topLeft[c] * (1 - deltaRH) * (1 - deltaRW)
                            + topRight[c] * (1 - deltaRH) * deltaRW
                            + bottomLeft[c] * (1 - deltaRW) * deltaRH
                            + bottomRight[c] * deltaRH * deltaRW;
                    }
                }
            }
        }
    }

    MAI_STATUS run() override {
        const Tensor* input = getInputTensor(0);
        Tensor* output = getOutputTensor(0);
        MAI_OP_RUN_FIRST_START
        const Tensor* size = getInputTensor(1);
        MAI_CHECK_NULL(input);
        MAI_CHECK_NULL(size);
        MAI_CHECK(input->dimSize() == 4, "ResizeBilinear can only support input with 4 dimensions but not %d", input->dimSize());
        MAI_CHECK(size->dimSize() == 1, "dimension of size must be 1 but not %d", size->dimSize());
        MAI_CHECK(size->elementSize() == 2, "element count of size must be 2 but not %d", size->elementSize());
        std::vector<shape_t> outputShape(4);
        int32 outputHeight = 0;
        int32 outputWidth = 0;
        if (size->dataType() ==  DT_INT32) {
            const int32* sizeData = size->data<int32>();
            outputHeight = sizeData[0];
            outputWidth = sizeData[1];
        } else if (size->dataType() == DT_INT64) {
            const int64* sizeData = size->data<int64>();
            outputHeight = static_cast<int32>(sizeData[0]);
            outputWidth = static_cast<int32>(sizeData[1]);
        }
        outputShape[input->n()] = input->dimN();
        outputShape[input->h()] = outputHeight;
        outputShape[input->w()] = outputWidth;
        outputShape[input->c()] = input->dimC();
        MAI_CHECK_NULL(output);
        output->resize(outputShape);

        MAI_OP_RUN_FIRST_END

        if (input->getDataFormat() == NHWC) {
            resizeNHWC(input->shape(), input->data<T>(), output->shape(), output->mutableData<T>());
        }

        return MAI_SUCCESS;
    }
private:
    bool mRunFirst;
};

void registerResizeBilinear() {
    MAI_REGISTER_OP((OpContext{.opType=RESIZE_BILINEAR,}), float, ResizeBilinear);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
