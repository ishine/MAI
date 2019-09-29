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

namespace {
template<class T, int32 rank>
class PadImpl{
public:
static void pad(const T* input, const std::vector<shape_t>& inputShape,
        T constantValue,
        const std::vector<int32>& paddings,
        T* output, const std::vector<shape_t>& outputShape);
};

template<class T>
class PadImpl<T, 4> {
public:
static void pad(const T* input, const std::vector<shape_t>& inputShape,
        T constantValue,
        const std::vector<int32>& paddings,
        T* output, const std::vector<shape_t>& outputShape) {
    // level 0
    shape_t paddingDim0Begin = paddings[0] * outputShape[1] * outputShape[2] * outputShape[3];
    shape_t paddingDim0End = paddings[1] * outputShape[1] * outputShape[2] * outputShape[3];
    for (shape_t n = 0; n < paddingDim0Begin; ++n) {
        *output++ = constantValue;
    }
    for (shape_t n = paddings[0]; n < outputShape[0] - paddings[0]; ++n) {
        // level 1
        shape_t paddingDim1Begin = paddings[2] * outputShape[2] * outputShape[3];
        shape_t paddingDim1End = paddings[3] * outputShape[2] * outputShape[3];
        for (shape_t h = 0; h < paddingDim1Begin; ++h) {
            *output++ = constantValue;
        }
        for (shape_t h = paddings[2]; h < outputShape[1] - paddings[2]; ++h) {
            // level 2
            shape_t paddingDim2Begin = paddings[4] * outputShape[3];
            shape_t paddingDim2End = paddings[5] * outputShape[3];
            for (shape_t w = 0; w < paddingDim2Begin; ++w) {
                *output++ = constantValue;
            }
            for (shape_t w = paddings[4]; w < outputShape[2] - paddings[4]; ++w) {
                // level 3
                shape_t paddingDim3Begin = paddings[6];
                shape_t paddingDim3End = paddings[7];
                for (shape_t c = 0; c < paddingDim3Begin; ++c) {
                    *output++ = constantValue;
                }
                shape_t inputOffset = offset4D(inputShape,
                        (n - paddings[0]), (h - paddings[2]), (w - paddings[4]), 0);

                memcpy(output, input + inputOffset, inputShape[3] * sizeof(T));
                output += inputShape[3];
                //for (shape_t c = paddingDim2Begin; w < outputShape[3] - paddingDim3End; ++c) {
                //    // level 3
                //    *output++ = input + offset4D(inputShape,
                //            (n - paddingDim0Begin), (h - paddingDim1Begin), (w - paddingDim2Begin), (c - paddingDim3Begin));
                //}
                for (shape_t c = 0; c < paddingDim3End; ++c) {
                    *output++ = constantValue;
                }
            }
            for (shape_t w = 0; w < paddingDim2End; ++w) {
                *output++ = constantValue;
            }
        }
        for (shape_t h = 0; h < paddingDim1End; ++h) {
            *output++ = constantValue;
        }
    }
    for (shape_t n = 0; n < paddingDim0End; ++n) {
        *output++ = constantValue;
    }
}
};

}

template<typename T>
class Pad : public Operator {
public:
    Pad() : mConstantValue(0) {}
    ~Pad() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    void setParam(Param* param) override {
        PadParam* padParam = reinterpret_cast<PadParam*>(param);
        if (padParam) {
            mConstantValue = static_cast<T>(padParam->constantValue);
            mPaddings = padParam->paddings;
            delete param;
        }
    }

    MAI_STATUS run() override {
        if (inputNames().size() > 1) {// paddings list is dynamic tensor
            MAI_CHECK(mPaddings.size() == 0, "Paddings is dynamic tensor but param is not empty");
            const Tensor* paddingTensor = getInputTensor(PADDINGS);
            MAI_CHECK_NULL(paddingTensor);
            mPaddings.resize(paddingTensor->elementSize());
            if (paddingTensor->dataType() == DT_INT32) {
                const int32* paddingData = paddingTensor->data<int32>();
                for (shape_t i = 0; i < paddingTensor->elementSize(); ++i) {
                    mPaddings[i] = paddingData[i];
                }
            } else if (paddingTensor->dataType() == DT_INT64) {
                const int64* paddingData = paddingTensor->data<int64>();
                for (shape_t i = 0; i < paddingTensor->elementSize(); ++i) {
                    mPaddings[i] = paddingData[i];
                }
            } else {
                MAI_ABORT("Unsupport padding data type : %s",
                        getNameFromDataType(paddingTensor->dataType()).c_str());
            }
        }
        const Tensor* input = getInputTensor(INPUT);
        MAI_CHECK_NULL(input);
        MAI_CHECK(mPaddings.size() == (input->dimSize() * 2), "Invalid paddings size:%d input dimSize:%d",
                mPaddings.size(), input->dimSize());
        //TODO: (gavinchen) for onnx if paddings[i] < 0, this means need to remove some value
        std::vector<shape_t> outputShape(input->shape());

        for (shape_t i = 0; i < input->dimSize(); ++i)  {
            outputShape[i] += (mPaddings[2 * i] + mPaddings[2 * i + 1]);
        }
        Tensor* output = getOutputTensor(OUTPUT);
        MAI_CHECK_NULL(output);
        output->resize(outputShape);
        T* outputData = output->mutableData<T>();
        if (input->dimSize() == 4) {
            PadImpl<T, 4>::pad(input->data<T>(), input->shape(), mConstantValue, mPaddings, outputData, outputShape);
        } else {
            MAI_ABORT("Unsupported dim size:%d", input->dimSize());
        }

        return MAI_SUCCESS;
    }
private:
    enum FLAG{INPUT, PADDINGS, OUTPUT = 0};
    T mConstantValue;
    std::vector<int32> mPaddings;
};

void registerPad() {
    MAI_REGISTER_OP((OpContext{.opType=PAD,}), float, Pad);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
