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
#include <string.h>
#include "core/OperatorRegister.h"
#include "util/MAIUtil.h"

namespace MAI {
namespace Op {
namespace CPU {

template<typename T>
class TransposeConv2d : public Operator {
public:
    TransposeConv2d() : mParam(NULL), mStrides(2), mRunFirst(true) {}
    ~TransposeConv2d() {
        MAI_DELETE_PTR(mParam);
    }

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    void setParam(Param* param) override {
        mParam = reinterpret_cast<TransposeConv2dParam*>(param);
        mStrides[0] = mParam->strides[1];
        mStrides[1] = mParam->strides[2];
    }

    MAI_STATUS run() override {
        MAI_OP_RUN_FIRST_START
        mOutputShape = getInputTensor(OUTPUT_SHAPE);
        mFilter = getInputTensor(FILTER);
        mInput = getInputTensor(INPUT);
        mOutput = getOutputTensor(OUTPUT);
        MAI_CHECK_NULL(mOutputShape);
        MAI_CHECK_NULL(mFilter);
        MAI_CHECK_NULL(mInput);
        MAI_CHECK_NULL(mOutput);
        MAI_CHECK_NULL(mParam);
        MAI_CHECK(mInput->shape().size() == 4, "Input shape must be 4-d");
        MAI_CHECK(mInput->getDataFormat() == NHWC, "Transpose Conv2d just support NHWC now");
        MAI_CHECK(mFilter->getDataFormat() == HWOI, "Transpose Conv2d filter just support HWOI now but not %s",
                getNameFromDataFormat(mFilter->getDataFormat()).c_str());
        MAI_CHECK(mInput->dimC() == mFilter->dimI(), "Unexpected dim");
        std::vector<shape_t> outputShape(mOutputShape->elementSize());
        const int32* outputShapeData = mOutputShape->data<int32>();
        MAI_CHECK_NULL(outputShapeData);
        for (int32 i = 0; i < outputShape.size(); ++i) {
            outputShape[i] = outputShapeData[i];
        }
        mOutput->resize(outputShape);
        MAI_OP_RUN_FIRST_END

        std::vector<int32> paddingTBLR = calcPaddings(
                mParam->paddingMode,
                {mFilter->dimH(), mFilter->dimW()});
        T* outputData = mOutput->mutableData<T>();
        memset(outputData, 0, mOutput->size());
        const T* inputData = mInput->data<T>();
        const T* filterData = mFilter->data<T>();
        const shape_t IBatch = mInput->dimN();
        const shape_t IHeight = mInput->dimH();
        const shape_t IWidth = mInput->dimW();
        const shape_t IChannel = mInput->dimC();

        const shape_t OBatch = mOutput->dimN();
        const shape_t OHeight = mOutput->dimH();
        const shape_t OWidth = mOutput->dimW();
        const shape_t OChannel = mOutput->dimC();

        const shape_t FH = mFilter->dimH();
        const shape_t FW = mFilter->dimW();
        const shape_t FI = mFilter->dimI();
        const shape_t FO = mFilter->dimO();

        for (shape_t b = 0; b < IBatch; ++b) {
            for (shape_t h = 0; h < IHeight; ++h) {
                for (shape_t w = 0; w < IWidth; ++w) {
                    for (shape_t c = 0; c < IChannel; ++c) {
                        const shape_t outHIndexBase = h * mStrides[0] - paddingTBLR[0];
                        const shape_t outWIndexBase = w * mStrides[1] - paddingTBLR[2];
                        for (shape_t fh = 0; fh < FH; ++fh) {
                            for (shape_t fw = 0; fw < FW; ++fw) {
                                for (shape_t oc = 0; oc < OChannel; ++oc) {
                                    const shape_t outHIndex = outHIndexBase + fh;
                                    const shape_t outWIndex = outWIndexBase + fw;
                                    if (outHIndex >= 0 && outHIndex < OHeight &&
                                            outWIndex >= 0 && outWIndex < OWidth) {
                                        T inputValue = inputData[offset4D(mInput->shape(), b, h, w, c)];
                                        T filterValue = filterData[offset4D(mFilter->shape(), fh, fw, oc, c)];
                                        outputData[offset4D(mOutput->shape(), b, outHIndex, outWIndex, oc)] += inputValue * filterValue;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return MAI_SUCCESS;
    }

private:
    enum FLAG {OUTPUT_SHAPE, FILTER, INPUT, OUTPUT = 0,};
    const Tensor* mOutputShape;
    const Tensor* mFilter;
    const Tensor* mInput;
    Tensor* mOutput;
    TransposeConv2dParam* mParam;
    std::vector<int32> mStrides;
    bool mRunFirst;
};

void registerTransposeConv2d() {
    MAI_REGISTER_OP((OpContext{.opType=TRANSPOSE_CONV2D,}), float, TransposeConv2d);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
