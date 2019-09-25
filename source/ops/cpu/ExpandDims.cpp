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

#include <limits>
#include "core/OperatorRegister.h"
#include "util/MAIUtil.h"

namespace MAI {
namespace Op {
namespace CPU {

template<typename T>
class ExpandDims : public Operator {
public:
    ExpandDims() : mAxis(std::numeric_limits<int32>::min()) {}
    ~ExpandDims() = default;

    MAI_STATUS init() override {
        const Tensor* input = getInputTensor(0);
        Tensor* output = getOutputTensor(0);
        MAI_CHECK_NULL(input);
        MAI_CHECK_NULL(output);
        MAI_CHECK(mAxis >= -(input->dimSize() + 1) && mAxis <= input->dimSize(), "Axis(%d) is not valid", mAxis);
        if (mAxis < 0) {
            mAxis = mAxis + input->dimSize() + 1;
        }
        std::vector<shape_t> outputShape(input->dimSize() + 1);
        for (int32 i = 0; i < input->dimSize() + 1; ++i) {
            if (i < mAxis) {
                outputShape[i] = input->dim(i);
            } else if (i == mAxis) {
                outputShape[i] = 1;
            } else {
                outputShape[i] = input->dim(i - 1);
            }
        }
        output->reuse(input);
        output->reshape(outputShape);
        return MAI_SUCCESS;
    }

    void setParam(Param* param) override {
        ExpandDimsParam* expandDimsParam = reinterpret_cast<ExpandDimsParam*>(param);
        if (expandDimsParam) {
            mAxis = expandDimsParam->axis;
        }
    }

    MAI_STATUS run() override {
        //const Tensor* input = getInputTensor(0);
        //Tensor* output = getOutputTensor(0);
        //MAI_CHECK_NULL(input);
        //MAI_CHECK_NULL(output);
        //MAI_CHECK(mAxis >= -(input->dimSize() + 1) && mAxis <= input->dimSize(), "Axis(%d) is not valid", mAxis);
        //if (mAxis < 0) {
        //    mAxis = mAxis + input->dimSize() + 1;
        //}
        //std::vector<shape_t> outputShape(input->dimSize() + 1);
        //for (int32 i = 0; i < input->dimSize() + 1; ++i) {
        //    if (i < mAxis) {
        //        outputShape[i] = input->dim(i);
        //    } else if (i == mAxis) {
        //        outputShape[i] = 1;
        //    } else {
        //        outputShape[i] = input->dim(i - 1);
        //    }
        //}
        //output->reuse(input);
        //output->reshape(outputShape);
        return MAI_SUCCESS;
    }
private:
    int32 mAxis;
};

void registerExpandDims() {
    MAI_REGISTER_OP((OpContext{.opType=EXPAND_DIMS,}), float, ExpandDims);
    MAI_REGISTER_OP((OpContext{.opType=EXPAND_DIMS,}), int32, ExpandDims);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
