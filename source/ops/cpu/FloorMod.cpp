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

#include <math.h>
#include "core/OperatorRegister.h"
#include "util/MAIUtil.h"
#include "Broadcast.h"

namespace MAI {
namespace Op {
namespace CPU {

template<class T>
class FloorMod : public Broadcast<T, T> {
public:
    MAI_STATUS onCommonCompute(const Tensor* inputA, const Tensor* inputB,
            Tensor* output) {
        ALOGI("FloorMod::onCommonCompute=============");
        const T* inputAData = inputA->data<T>();
        const T* inputBData = inputB->data<T>();
        T* outputData = output->mutableData<T>();
        #pragma omp parallel for
        for (int32 i = 0; i < output->elementSize(); ++i) {
            outputData[i] = inputAData[i] - floor(inputAData[i] / inputBData[i]) * inputBData[i];
        }
        return MAI_SUCCESS;
    }

    MAI_STATUS onScalarCompute(const Tensor* input, const T inputScalar,
            Tensor* output, bool convertInput) {
        const T* inputData = input->data<T>();
        T* outputData = output->mutableData<T>();
        if (convertInput) {
            #pragma omp parallel for
            for (int32 i = 0; i < input->elementSize(); ++i) {
                outputData[i] = inputScalar - floor(inputScalar / inputData[i]) * inputData[i];
            }
        } else {
            #pragma omp parallel for
            for (int32 i = 0; i < input->elementSize(); ++i) {
                outputData[i] = inputData[i] - floor(inputData[i] / inputScalar) * inputScalar;
            }
        }
        return MAI_SUCCESS;
    }

    MAI_STATUS onBroadcastCompute() {
        auto addFunc = [](const T* x, const T* y, T* o) {
            *o = *x - floor(*x / *y) * *y;
        };
        return this->broadcastCompute(addFunc);
    }
};

void registerFloorMod() {
    MAI_REGISTER_OP((OpContextBuilder().setOperatorType(FLOOR_MOD).build()), float, FloorMod);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
