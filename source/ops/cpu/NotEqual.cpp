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
#include "Broadcast.h"

namespace MAI {
namespace Op {
namespace CPU {

template<class T>
class NotEqual : public Broadcast<T, int8> {
public:
    MAI_STATUS onCommonCompute(const Tensor* inputA, const Tensor* inputB,
            Tensor* output) {
        const T* inputAData = inputA->data<T>();
        const T* inputBData = inputB->data<T>();
        int8* outputData = output->mutableData<int8>();
        #pragma omp parallel for
        for (int32 i = 0; i < output->elementSize(); ++i) {
            outputData[i] = inputAData[i] != inputBData[i] ? 1 : 0;
        }
        return MAI_SUCCESS;
    }

    MAI_STATUS onScalarCompute(const Tensor* input, const T inputScalar,
            Tensor* output, bool convertInput) {
        const T* inputData = input->data<T>();
        int8* outputData = output->mutableData<int8>();
        #pragma omp parallel for
        for (int32 i = 0; i < input->elementSize(); ++i) {
            outputData[i] = inputData[i] != inputScalar ? 1 : 0;
            ALOGI("outputData i=%d, o=%d, inputData=%f, inputScalar:%f", i, outputData[i], inputData[i], inputScalar);
        }
        return MAI_SUCCESS;
    }

    MAI_STATUS onBroadcastCompute() {
        auto addFunc = [](const T* x, const T* y, int8* o) {
            *o = *x != *y ? 1 : 0;
        };
        return this->broadcastCompute(addFunc);
    }
};

void registerNotEqual() {
    MAI_REGISTER_OP((OpContextBuilder().setOperatorType(NOT_EQUAL).build()), float, NotEqual);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
