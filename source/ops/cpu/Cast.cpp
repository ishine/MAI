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

#define CAST_TYPE(DATA_TYPE) \
    if (outputDataType == DATA_TYPE) {                                                  \
        auto outputData = output->mutableData<EnumToDataType<DATA_TYPE>::Type>();       \
        for (shape_t i = 0; i < output->elementSize(); ++i)  {                          \
            outputData[i] = static_cast<EnumToDataType<DATA_TYPE>::Type>(inputData[i]); \
        }                                                                               \
        return MAI_SUCCESS;                                                             \
    }

template<typename T>
class Cast : public Operator {
public:
    Cast() = default;
    ~Cast() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    MAI_STATUS run() override {
        const Tensor* input = getInputTensor(0);
        Tensor* output = getOutputTensor(0);
        MAI_CHECK_NULL(input);
        MAI_CHECK_NULL(output);
        output->resize(input->shape());
        const T* inputData = input->data<T>();
        DataType outputDataType = output->dataType();
        CAST_TYPE(DT_FLOAT);
        CAST_TYPE(DT_INT8);
        CAST_TYPE(DT_INT32);
        CAST_TYPE(DT_UINT8);
        return MAI_FAILED;
    }
};

void registerCast() {
    MAI_REGISTER_OP((OpContext{.opType=CAST,}), int32, Cast);
    MAI_REGISTER_OP((OpContext{.opType=CAST,}), int64, Cast);
    MAI_REGISTER_OP((OpContext{.opType=CAST,}), float, Cast);
}

} // namespace CPU
} // namespace Op
} // namespace MAI