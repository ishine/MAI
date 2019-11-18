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
class CRelu : public Operator {
public:
    CRelu() : mRunFirst(true) {}
    ~CRelu() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    MAI_STATUS run() override {
        const Tensor* input = getInputTensor(0);
        Tensor* output = getOutputTensor(0);

        MAI_OP_RUN_FIRST_START
        MAI_CHECK_NULL(input);
        MAI_CHECK_NULL(output);
        std::vector<shape_t> outputShape(input->shape());
        outputShape[outputShape.size() - 1] *= 2;
        output->resize(outputShape);
        MAI_OP_RUN_FIRST_END

        const T* inputData = input->data<T>();
        T* outputData = output->mutableData<T>();
        shape_t outerSize = 1;
        for (int32 i = 0; i < output->dimSize() - 1; ++i) {
            outerSize *= output->dim(i);
        }
        shape_t innerSize = output->dim(output->dimSize() - 1) / 2;
        for (shape_t i = 0; i < outerSize; ++i) {
            for (shape_t in = 0; in < innerSize; ++in) {
                outputData[i * innerSize * 2 + in] = std::max(inputData[i * innerSize + in], static_cast<T>(0));
            }
            for (shape_t in = 0; in < innerSize; ++in) {
                outputData[i * innerSize * 2 + innerSize + in] = std::max(-inputData[i * innerSize + in], static_cast<T>(0));
            }
        }
        return MAI_SUCCESS;
    }
private:
    bool mRunFirst;
};

void registerCRelu() {
    MAI_REGISTER_OP((OpContextBuilder().setOperatorType(CRELU).build()), float, CRelu);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
