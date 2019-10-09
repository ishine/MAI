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
class Softmax : public Operator {
public:
    Softmax() : mAxis(-1), mBeta(1.f), mRunFirst(true) {
    }
    ~Softmax() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    void setParam(Param* param) override {
        SoftmaxParam* softmaxParam = reinterpret_cast<SoftmaxParam*>(param);
        if (softmaxParam) {
            mBeta = softmaxParam->beta;
            mAxis = softmaxParam->axis;
        }
    }

    void softmax2D(const T* input, int32 batch, int32 inSize, T* output) {
        for (int32 b = 0; b < batch; ++b) {
            //1. find the max value
            T max = input[0];
            for (int32 i = 1; i < inSize; ++i) {
                if (max < input[i]) {
                    max = input[i];
                }
            }

            //2. calculate sum of a batch
            T sum = 0.f;
            for (int32 i = 0; i < inSize; ++i) {
                output[i] = std::exp((input[i] - max) * mBeta);
                sum += output[i];
            }

            //3. calculate tf.exp(logits) / sum
            T reciprocalSum = 1.f / sum;
            for (int32 i = 0; i < inSize; ++i) {
                output[i] *= reciprocalSum;
            }
            input += inSize;
            output += inSize;
        }
    }

    MAI_STATUS run() override {
        const Tensor* input = getInputTensor(0);
        Tensor* output = getOutputTensor(0);

        MAI_OP_RUN_FIRST_START
        if (mAxis < 0) {
            mAxis += input->shape().size();
        }
        MAI_CHECK_NULL(input);
        MAI_CHECK_NULL(output);
        output->resize(input->shape());
        MAI_OP_RUN_FIRST_END

        const T* inputData = input->data<T>();
        T* outputData = output->mutableData<T>();

        if (input->shape().size() == 1) {
            MAI_CHECK((mAxis == 0), "axis must be 0 when rank of input is 1, but not : %d", mAxis);

            softmax2D(inputData, 1, input->dim(0), outputData);
        } else if (input->shape().size() == 2) {
            MAI_CHECK((mAxis == 0 || mAxis == 1), "axis must be 0 or 1 when rank of input is 2, \
                    but not : %d", mAxis);
            softmax2D(inputData, input->dim(0), input->dim(1), outputData);
        } else {
            MAI_ABORT("Unsupport shape size");
        }

        return MAI_SUCCESS;
    }

private:
    int32 mAxis;// TODO: gavinchen axis is not support now
    float mBeta;
    bool mRunFirst;
};

void registerSoftmax() {
    MAI_REGISTER_OP((OpContext{.opType=SOFTMAX,}), float, Softmax);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
