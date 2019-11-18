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

#ifdef MAI_NEON_ENABLED
#include "neon/TransposeNeon.h"
#else
#include "ref/TransposeRef.h"
#endif

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
#ifdef MAI_NEON_ENABLED
        if (input->dimSize() == 2) {
            if (input->dataType() == DT_INT32) {
                NEON::Transpose<int32, 2>::transpose(input->shape(), reinterpret_cast<const int32*>(inputData),
                        permData, output->shape(), reinterpret_cast<int32*>(outputData), sizeofElement);
                return MAI_SUCCESS;
            } else if (input->dataType() == DT_FLOAT) {
                NEON::Transpose<float, 2>::transpose(input->shape(), reinterpret_cast<const float*>(inputData),
                        permData, output->shape(), reinterpret_cast<float*>(outputData), sizeofElement);
                return MAI_SUCCESS;
            }
        } else if (input->dimSize() == 4) {
            // from NHWC -> NCHW
            if (permData[0] == 0 && permData[1] == 3 && permData[2] == 1 && permData[3] == 2) {
                if (input->dataType() == DT_INT32) {
                    shape_t imageSize = input->dim(1) * input->dim(2);
                    for (shape_t b = 0; b < input->dim(0); ++b) {
                        const int32* batchInputData = reinterpret_cast<const int32*>(inputData) + b * imageSize * input->dim(3);
                        int32* batchOutputData = reinterpret_cast<int32*>(outputData) + b * imageSize * input->dim(3);

                        NEON::Transpose<int32, 2>::transpose({imageSize, input->dim(3)}, batchInputData,
                                {}/*Not used*/, {input->dim(3), imageSize}, batchOutputData, sizeofElement/*Not used*/);
                        return MAI_SUCCESS;
                    }
                } else if (input->dataType() == DT_FLOAT) {
                    shape_t imageSize = input->dim(1) * input->dim(2);
                    for (shape_t b = 0; b < input->dim(0); ++b) {
                        const float* batchInputData = reinterpret_cast<const float*>(inputData) + b * imageSize * input->dim(3);
                        float* batchOutputData = reinterpret_cast<float*>(outputData) + b * imageSize * input->dim(3);

                        NEON::Transpose<float, 2>::transpose({imageSize, input->dim(3)}, batchInputData,
                                {}/*Not used*/, {input->dim(3), imageSize}, batchOutputData, sizeofElement/*Not used*/);
                        return MAI_SUCCESS;
                    }
                }
            } else if (permData[0] == 0 && permData[1] == 2 && permData[2] == 3 && permData[3] == 1) {// from NCHW -> NHWC
                if (input->dataType() == DT_INT32) {
                    shape_t imageSize = input->dim(2) * input->dim(3);
                    for (shape_t b = 0; b < input->dim(0); ++b) {
                        const int32* batchInputData = reinterpret_cast<const int32*>(inputData) + b * imageSize * input->dim(1);
                        int32* batchOutputData = reinterpret_cast<int32*>(outputData) + b * imageSize * input->dim(1);

                        NEON::Transpose<int32, 2>::transpose({input->dim(1), imageSize}, batchInputData,
                                {}/*Not used*/, {imageSize, input->dim(1)}, batchOutputData, sizeofElement/*Not used*/);
                        return MAI_SUCCESS;
                    }
                } else if (input->dataType() == DT_FLOAT) {
                    shape_t imageSize = input->dim(2) * input->dim(3);
                    for (shape_t b = 0; b < input->dim(0); ++b) {
                        const float* batchInputData = reinterpret_cast<const float*>(inputData) + b * imageSize * input->dim(1);
                        float* batchOutputData = reinterpret_cast<float*>(outputData) + b * imageSize * input->dim(1);

                        NEON::Transpose<float, 2>::transpose({input->dim(1), imageSize}, batchInputData,
                                {}/*Not used*/, {imageSize, input->dim(1)}, batchOutputData, sizeofElement/*Not used*/);
                        return MAI_SUCCESS;
                    }
                }
            }
        }
        NEON::Transpose<uint8, MAI_DYNAMIC_DIM>::transpose(input->shape(), inputData,
                permData, output->shape(), outputData, sizeofElement);
#else
        Ref::Transpose<uint8, MAI_DYNAMIC_DIM>::transpose(input->shape(), inputData,
                permData, output->shape(), outputData, sizeofElement);
#endif
        return MAI_SUCCESS;
    }
private:
    enum FLAG {INPUT, PERM, OUTPUT = 0};
    bool mRunFirst;
};

void registerTranspose() {
    MAI_REGISTER_OP((OpContextBuilder().setOperatorType(TRANSPOSE).build()), Transpose);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
