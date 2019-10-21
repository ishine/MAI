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
#include <omp.h>

namespace MAI {
namespace Op {
namespace CPU {

template<typename T>
class Relu6 : public Operator {
public:
    Relu6() : mRunFirst(true) {}
    ~Relu6() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    MAI_STATUS run() override {
        const Tensor* input = getInputTensor(0);
        Tensor* output = getOutputTensor(0);

        MAI_OP_RUN_FIRST_START
        MAI_CHECK_NULL(input);
        MAI_CHECK(output != NULL, "cannot get output tensor size:%s", outputNames()[0].c_str());
        MAI_CHECK_NULL(output);
        output->resize(input->shape());
        MAI_OP_RUN_FIRST_END

        const T* inputData = input->data<T>();
        T* outputData = output->mutableData<T>();
        #pragma omp parallel for
        for (shape_t i = 0; i < input->elementSize(); ++i) {
            //ALOGI("i = %d, I am Thread %d\n", i, omp_get_num_threads());
            outputData[i] = std::min(std::max(inputData[i], static_cast<T>(0)), static_cast<T>(6));
        }
        return MAI_SUCCESS;
    }
private:
    bool mRunFirst;
};

void registerRelu6() {
    MAI_REGISTER_OP((OpContext{.opType=RELU6,}), float, Relu6);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
