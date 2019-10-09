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

#define CAST_TYPE1(DATA_TYPE) \
    if (outputDataType == DATA_TYPE) {                                                  \
        auto outputData = output->mutableData<EnumToDataType<DATA_TYPE>::Type>();       \
        for (shape_t i = 0; i < output->elementSize(); ++i)  {                          \
            outputData[i] = static_cast<EnumToDataType<DATA_TYPE>::Type>(inputData[i]); \
        }                                                                               \
    }
        //return MAI_SUCCESS;                                                             \
    }

#define CAST_TYPE(SRC_DATA_TYPE, DST_DATA_TYPE)                             \
    if (input->dataType() == DataTypeToEnum<SRC_DATA_TYPE>::value           \
            && output->dataType() == DataTypeToEnum<DST_DATA_TYPE>::value) {\
        cast<SRC_DATA_TYPE, DST_DATA_TYPE>(input, output);                  \
        return MAI_SUCCESS;                                                 \
    }

template<class SrcType, class DstType>
void cast(const Tensor* input, Tensor* output) {
    const SrcType* inputData = input->data<SrcType>();
    DstType* outputData = output->mutableData<DstType>();
    for (shape_t i = 0; i < output->elementSize(); ++i) {
        outputData[i] = static_cast<DstType>(inputData[i]);
    }
}

class Cast : public Operator {
public:
    Cast() : mRunFirst(true) {}
    ~Cast() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    MAI_STATUS run() override {
        const Tensor* input = getInputTensor(0);
        Tensor* output = getOutputTensor(0);
        MAI_OP_RUN_FIRST_START
        MAI_CHECK_NULL(input);
        MAI_CHECK_NULL(output);
        output->resize(input->shape());
        MAI_OP_RUN_FIRST_END

        CAST_TYPE(float, int8);
        CAST_TYPE(float, int32);
        CAST_TYPE(float, int64);

        CAST_TYPE(int8, int8);
        CAST_TYPE(int8, int32);
        CAST_TYPE(int8, int64);
        CAST_TYPE(int8, float);

        CAST_TYPE(int32, int8);
        CAST_TYPE(int32, int32);
        CAST_TYPE(int32, int64);
        CAST_TYPE(int32, float);

        CAST_TYPE(int64, int8);
        CAST_TYPE(int64, int32);
        CAST_TYPE(int64, int64);
        CAST_TYPE(int64, float);
        MAI_ABORT("Unsupported now(%s --> %s)", getNameFromDataType(input->dataType()).c_str(),
                getNameFromDataType(output->dataType()).c_str());
        return MAI_FAILED;
    }
private:
    bool mRunFirst;
};

void registerCast() {
    MAI_REGISTER_OP((OpContext{.opType=CAST,}), Cast);
    //MAI_REGISTER_OP((OpContext{.opType=CAST,}), int64, Cast);
    //MAI_REGISTER_OP((OpContext{.opType=CAST,}), float, Cast);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
