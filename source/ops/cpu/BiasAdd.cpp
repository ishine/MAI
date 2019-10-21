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
class BiasAdd : public Operator {
public:
    BiasAdd() : mRunFirst(true) {}
    ~BiasAdd() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    void setParam(Param* param) override {
        MAI_ABORT("Unsupported setParam for BiasAdd");
    }

    static void biasAddNHWC(const T* input,
            const std::vector<shape_t>& inputShape,
            const T* bias,
            const std::vector<shape_t>& biasShape,
            T* output,
            const std::vector<shape_t>& outputShape) {
        shape_t NHW = inputShape[0] * inputShape[1] * inputShape[2];
        shape_t C = inputShape[3];
        for (shape_t i = 0; i < NHW; ++i) {
            for (shape_t c = 0; c < C; ++c) {
                output[i * C + c] = input[i * C + c] + bias[c];
            }
        }
    }

    static void biasAddNCHW(const T* input,
            const std::vector<shape_t>& inputShape,
            const T* bias,
            const std::vector<shape_t>& biasShape,
            T* output,
            const std::vector<shape_t>& outputShape) {
        shape_t batchInnerSize = inputShape[1] * inputShape[2] * inputShape[3];
        shape_t channelInnerSize = inputShape[2] * inputShape[3];
        shape_t N = inputShape[0];
        shape_t C = inputShape[1];
        shape_t HW = channelInnerSize;

        for (shape_t n = 0; n < N; ++n) {
            for (shape_t c = 0; c < C; ++c) {
                for (shape_t hw = 0; hw < HW; ++hw) {
                    output[n * batchInnerSize + c * channelInnerSize + hw] =
                        input[n * batchInnerSize + c * channelInnerSize + hw] + bias[c];
                }
            }
        }
    }

    MAI_STATUS run() override {
        MAI_OP_RUN_FIRST_START
        mInput = getInputTensor(INPUT);
        mBias = getInputTensor(BIAS);
        mOutput = getOutputTensor(OUTPUT);
        MAI_CHECK_NULL(mInput);
        MAI_CHECK_NULL(mBias);
        MAI_CHECK_NULL(mOutput);
        mOutput->resize(mInput->shape());

        if (mInput->getDataFormat() == NHWC) {
            mFunction = biasAddNHWC;
        } else if (mInput->getDataFormat() == NCHW) {
            mFunction = biasAddNCHW;
        } else {
            MAI_CHECK(false, "Unsupported data format: %d", mInput->getDataFormat());
        }
        MAI_OP_RUN_FIRST_END

        mFunction(mInput->data<T>(), mInput->shape(), mBias->data<T>(), mBias->shape(), mOutput->mutableData<T>(), mOutput->shape());
        return MAI_SUCCESS;
    }

private:
    enum FLAG {INPUT, BIAS, OUTPUT = 0,};
    const Tensor* mInput;
    const Tensor* mBias;
    Tensor* mOutput;
    std::function<void(const T*, const std::vector<shape_t>&,
            const T*, const std::vector<shape_t>&,
            T*, const std::vector<shape_t>&)> mFunction;
    bool mRunFirst;
};

void registerBiasAdd() {
    MAI_REGISTER_OP((OpContext{.opType=BIAS_ADD,}), float, BiasAdd);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
