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

#include <cstring>
#include "core/OperatorRegister.h"
#include "util/MAIUtil.h"
#include "ref/TransposeRef.h"

namespace MAI {
namespace Op {
namespace CPU {

class Transpose : public Operator {
public:
    Transpose() : mRunFirst(true) {}
    ~Transpose() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    MAI_STATUS run() override {
        const Tensor* input = getInputTensor(INPUT);
        const Tensor* perm = getInputTensor(PERM);
        Tensor* output = getOutputTensor(OUTPUT);

        const int32* permData = perm->data<int32>();
        MAI_OP_RUN_FIRST_START
        MAI_CHECK_NULL(input);
        MAI_CHECK_NULL(perm);
        MAI_CHECK_NULL(output);
        MAI_CHECK(input->dimSize() == perm->elementSize(),
                "rank of input(%d) must be equal to perm data size(%d)", input->dimSize(), perm->elementSize());
        std::vector<shape_t> outputShape(input->dimSize());
        for (shape_t i = 0; i < input->dimSize(); ++i) {
            outputShape[i] = input->dim(permData[i]);
        }
        output->resize(outputShape);
        MAI_OP_RUN_FIRST_END

        const uint8* inputData = input->data<uint8>();
        const int32 sizeofElement = input->size() / input->elementSize();
        uint8* outputData = output->mutableData<uint8>();
        Ref::Transpose<uint8, MAI_DYNAMIC_DIM>::transpose(input->shape(), inputData,
                permData, output->shape(), outputData, sizeofElement);
        return MAI_FAILED;
    }
private:
    enum FLAG {INPUT, PERM, OUTPUT = 0};
    bool mRunFirst;
};

void registerTranspose() {
    MAI_REGISTER_OP((OpContext{.opType=TRANSPOSE,}), Transpose);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
