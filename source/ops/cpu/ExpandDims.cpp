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

class ExpandDims : public Operator {
public:
    ExpandDims() = default;
    ~ExpandDims() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    void setParam(Param* param) override {
        ExpandDimsParam* expandDimsParam = reinterpret_cast<ExpandDimsParam*>(param);
        if (expandDimsParam) {
            mAxes = expandDimsParam->axes;
            delete param;
        }
    }

    MAI_STATUS run() override {
        const Tensor* input = getInputTensor(0);
        Tensor* output = getOutputTensor(0);
        MAI_CHECK_NULL(input);
        MAI_CHECK_NULL(output);
        MAI_CHECK(mAxes.size() > 0, "At least one axis need to be sepecified");
        if (mAxes.size() == 1) {
            int32 axis = mAxes[0];
            MAI_CHECK(axis >= -(input->dimSize() + 1) && axis <= input->dimSize(), "Axis(%d) is not valid", axis);
            if (mAxes[0] < 0) {
                mAxes[0] = mAxes[0] + input->dimSize() + 1;
            }
        }
        for (int32 i = 0; i < mAxes.size(); ++i) {
            int32 axis = mAxes[i];
            MAI_CHECK(axis >= 0, "Onnx operator can only support positive axis");
        }
        std::vector<shape_t> outputShape(input->shape());
        for (int32 i = 0; i < mAxes.size(); ++i) {
            outputShape.insert(outputShape.begin() + mAxes[i], 1);
        }
        output->reuse(input);
        output->reshape(outputShape);
        return MAI_SUCCESS;
    }
private:
    std::vector<int32> mAxes;
};

void registerExpandDims() {
    MAI_REGISTER_OP((OpContext{.opType=EXPAND_DIMS,}), ExpandDims);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
