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
class Floor : public Operator {
public:
    Floor() : mRunFirst(true) {}
    ~Floor() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    MAI_STATUS run() override {
        const Tensor* input = getInputTensor(0);
        // TODO (gavinchen) check the datatype of input is half, bfloat16, float32, float64
        Tensor* output = getOutputTensor(0);

        MAI_OP_RUN_FIRST_START
        MAI_CHECK_NULL(input);
        MAI_CHECK_NULL(output);
        output->resize(input->shape());
        MAI_OP_RUN_FIRST_END

        const T* inputData = input->data<T>();
        T* outputData = output->mutableData<T>();
        for (shape_t i = 0; i < input->elementSize(); ++i) {
            outputData[i] = std::floor(inputData[i]);
        }
        return MAI_FAILED;
    }
private:
    bool mRunFirst;
};

void registerFloor() {
    MAI_REGISTER_OP((OpContext{.opType=FLOOR,}), float, Floor);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
