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

#include "core/OperatorRegister.h"
#include "util/MAIUtil.h"

namespace MAI {
namespace Op {
namespace CPU {

template<typename T>
class LeakyRelu : public Operator {
public:
    LeakyRelu() : mParam(NULL), mRunFirst(true) {}
    ~LeakyRelu() {
        MAI_DELETE_PTR(mParam);
    }
    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    void setParam(Param* param) override {
        ALOGI("setParam");
        mParam = reinterpret_cast<LeakyReluParam*>(param);
    }

    MAI_STATUS run() override {
        const Tensor* input = getInputTensor(0);
        Tensor* output = getOutputTensor(0);

        ALOGI("1");
        MAI_OP_RUN_FIRST_START
        MAI_CHECK_NULL(input);
        MAI_CHECK_NULL(output);
        MAI_CHECK_NULL(mParam);
        ALOGI("2");
        output->resize(input->shape());
        MAI_OP_RUN_FIRST_END

        const T* inputData = input->data<T>();
        T* outputData = output->mutableData<T>();
        ALOGI("3:%f", mParam->alpha);
        for (shape_t i = 0; i < output->elementSize(); ++i) {
            outputData[i] = inputData[i] >= 0 ? inputData[i] : inputData[i] * mParam->alpha;
        }
        return MAI_SUCCESS;
    }
private:
    bool mRunFirst;
    LeakyReluParam* mParam;
};

void registerLeakyRelu() {
    MAI_REGISTER_OP((OpContext{.opType=LEAKY_RELU,}), float, LeakyRelu);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
