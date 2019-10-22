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
class TransposeConv2d : public Operator {
public:
    TransposeConv2d() : mParam(NULL), mRunFirst(true) {}
    ~TransposeConv2d() {
        MAI_DELETE_PTR(mParam);
    }

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    void setParam(Param* param) override {
        //mParam = reinterpret_cast<TransposeConv2dParam*>(param);
    }

    MAI_STATUS run() override {
        MAI_OP_RUN_FIRST_START
        mOutputShape = getInputTensor(OUTPUT_SHAPE);
        mFilter = getInputTensor(FILTER);
        mInput = getInputTensor(INPUT);
        mOutput = getOutputTensor(OUTPUT);
        MAI_CHECK_NULL(mOutputShape);
        MAI_CHECK_NULL(mFilter);
        MAI_CHECK_NULL(mInput);
        MAI_CHECK_NULL(mOutput);
        MAI_CHECK_NULL(mParam);
        MAI_CHECK(mInput->shape().size() == 4, "Input shape must be 4-d");
        MAI_OP_RUN_FIRST_END

        return MAI_SUCCESS;
    }

private:
    enum FLAG {OUTPUT_SHAPE, FILTER, INPUT, OUTPUT = 0,};
    const Tensor* mOutputShape;
    const Tensor* mFilter;
    const Tensor* mInput;
    Tensor* mOutput;
    Param* mParam;
    bool mRunFirst;
};

void registerTransposeConv2d() {
    MAI_REGISTER_OP((OpContext{.opType=TRANSPOSE_CONV2D,}), float, TransposeConv2d);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
