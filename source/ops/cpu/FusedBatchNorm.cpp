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

#include <cmath>
#include "core/OperatorRegister.h"
#include "util/MAIUtil.h"

namespace MAI {
namespace Op {
namespace CPU {

template<typename T>
class FusedBatchNorm : public Operator {
public:
    FusedBatchNorm() : mEpsilon(0.001f), mRunFirst(true) {
    }

    ~FusedBatchNorm() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    static void fusedBatchNormNHWC(const T* input, const std::vector<shape_t>& inputShape,
            const std::vector<T>& scale, const std::vector<T>& offset, T* output) {
        int idx = 0;
        // use openmp will cost more time?
        //#pragma omp parallel for collapse(2)
        for (shape_t b = 0; b < inputShape[0]; ++b) {
            for (shape_t h = 0; h < inputShape[1]; ++h) {
                for (shape_t w = 0; w < inputShape[2]; ++w) {
                    for (shape_t c = 0; c < inputShape[3]; ++c) {
                        output[idx] = input[idx] * scale[c] + offset[c];
                        idx++;
                    }
                }
            }
        }
    }

    static void fusedBatchNormNCHW(const T* input, const std::vector<shape_t>& inputShape,
            const std::vector<T>& scale, const std::vector<T>& offset, T* output) {
        int idx = 0;
        // use openmp will cost more time?
        //#pragma omp parallel for collapse(2)
        for (shape_t b = 0; b < inputShape[0]; ++b) {
            for (shape_t c = 0; c < inputShape[1]; ++c) {
                for (shape_t h = 0; h < inputShape[2]; ++h) {
                    for (shape_t w = 0; w < inputShape[3]; ++w) {
                        output[idx] = input[idx] * scale[c] + offset[c];
                        idx++;
                    }
                }
            }
        }
    }

    void setParam(Param* param) override {
        FusedBatchNormParam* fParam = reinterpret_cast<FusedBatchNormParam*>(param);
        mEpsilon = fParam->epsilon;
    }

    MAI_STATUS run() override {
        MAI_OP_RUN_FIRST_START
        mInput = getInputTensor(INPUT);
        const Tensor* scale = getInputTensor(SCALE);
        const Tensor* offset = getInputTensor(OFFSET);
        const Tensor* mean = getInputTensor(MEAN);
        const Tensor* var = getInputTensor(VAR);

        MAI_CHECK(mInput->dimSize() ==  4, "rank of input must be 4, but not %d", mInput->dimSize());
        MAI_CHECK(scale->dimSize() ==  1, "rank of input must be 1, but not %d", scale->dimSize());
        MAI_CHECK(offset->dimSize() ==  1, "rank of input must be 1, but not %d", offset->dimSize());
        MAI_CHECK(mean->dimSize() ==  1, "rank of input must be 1, but not %d", mean->dimSize());
        MAI_CHECK(var->dimSize() ==  1, "rank of input must be 1, but not %d", var->dimSize());

        mOutput = getOutputTensor(OUTPUT);
        mOutput->resize(mInput->shape());

        shape_t channel = 0;
        if (mInput->getDataFormat() == NHWC) {
            mFunction = fusedBatchNormNHWC;
            channel = mInput->dim(3);
        } else if (mInput->getDataFormat() == NCHW) {
            mFunction = fusedBatchNormNCHW;
            channel = mInput->dim(1);
        } else {
            MAI_CHECK(false, "Unsupported data format: %d", mInput->getDataFormat());
        }

        mNewScale.resize(channel);
        mNewOffset.resize(channel);

        const T* scaleData = scale->data<T>();
        const T* offsetData = offset->data<T>();
        const T* meanData = mean->data<T>();
        const T* varData = var->data<T>();

        // z = gamma * (y - mean) / sqrt(variance + epsilon) + beta
        for (shape_t c = 0; c < channel; ++c) {
            mNewScale[c] = scaleData[c] / std::sqrt(varData[c] + mEpsilon);
            mNewOffset[c] = offsetData[c] - mNewScale[c] * meanData[c];
        }
        MAI_OP_RUN_FIRST_END

        mFunction(mInput->data<T>(), mInput->shape(), mNewScale,
            mNewOffset, mOutput->mutableData<T>());
        return MAI_SUCCESS;
    }
private:
    enum Flags {INPUT = 0, SCALE, OFFSET, MEAN, VAR = 4, OUTPUT = 0};
    std::function<void(const T*, const std::vector<shape_t>&, const std::vector<T>&,
                const std::vector<T>&, T*)> mFunction;
    float mEpsilon;
    std::vector<T> mNewScale;
    std::vector<T> mNewOffset;
    const Tensor* mInput;
    Tensor* mOutput;
    bool mRunFirst;
};

void registerFusedBatchNorm() {
    MAI_REGISTER_OP((OpContext{.opType=FUSED_BATCH_NORM,}), float, FusedBatchNorm);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
