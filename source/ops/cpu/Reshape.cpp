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

#include <algorithm>
#include "core/OperatorRegister.h"
#include "util/MAIUtil.h"

namespace MAI {
namespace Op {
namespace CPU {

template<typename T>
class Reshape : public Operator {
public:
    Reshape() : mRunFirst(true) {}
    ~Reshape() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    MAI_STATUS run() override {
        MAI_OP_RUN_FIRST_START
        const Tensor* inputPtr = getInputTensor(0);
        const Tensor* shapePtr = getInputTensor(1);
        Tensor* outputPtr = getOutputTensor(0);

        MAI_CHECK_NULL(inputPtr);
        MAI_CHECK_NULL(shapePtr);
        std::vector<shape_t> outputShape(shapePtr->elementSize());
        std::vector<shape_t> shape(shapePtr->elementSize());
        if (shapePtr->dataType() == DT_INT32) {
            auto shapeData = shapePtr->data<int32>();
            for (shape_t i = 0; i < shapePtr->elementSize(); ++i) {
                shape[i] = shapeData[i];
            }
        } else {
            auto shapeData = shapePtr->data<int64>();
            for (shape_t i = 0; i < shapePtr->elementSize(); ++i) {
                shape[i] = shapeData[i];
            }
        }

        int64 unknownDims = -1;
        uint8 index = 0;
        std::for_each(shape.begin(), shape.end(), [&](shape_t v){
            if (v == -1) {
                if(unknownDims != -1) {
                    MAI_CHECK(0, "Only one dim can be dynamic");
                    //error
                } else {
                    unknownDims = index;
                }
            }
            outputShape[index] = unknownDims == index ? 0 : v;
            ++index;
        });
        if (unknownDims != -1) {
            shape_t shapeSize = shapeToSizeSkipDim(outputShape, unknownDims);
            shape_t dynamicDim = inputPtr->elementSize() / shapeSize;
            MAI_CHECK((dynamicDim * shapeSize) == inputPtr->elementSize(), "Error shape");
            outputShape[unknownDims] = dynamicDim;

        } else {
            shape_t shapeSize = shapeToSize(outputShape);
            MAI_CHECK((inputPtr->elementSize() == shapeSize), "Error shape");
        }

        outputPtr->reuse(inputPtr);
        outputPtr->reshape(outputShape);
        MAI_OP_RUN_FIRST_END

        return MAI_SUCCESS;
    }
private:
    bool mRunFirst;
};

void registerReshape() {
    MAI_REGISTER_OP((OpContext{.opType=RESHAPE,}), float, Reshape);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
