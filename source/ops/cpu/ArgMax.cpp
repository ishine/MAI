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
class ArgMax : public Operator {
public:
    ArgMax() : mAxis(0), mOuterSize(1), mInnerSize(1), mParam(NULL), mRunFirst(true) {}
    ~ArgMax() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    void setParam(Param* param) override {
        mParam = reinterpret_cast<ArgMaxParam*>(param);
    }

    Param* getParam() override {
        return mParam;
    }

    template<typename T1, typename T2>
    static void argMax(const T1* inputData, T2* outputData,
            int32 axisSize, int32 outerSize, int32 innerSize) {
        for (int32 outer = 0; outer < outerSize; ++outer) {
            for (int32 inner = 0; inner < innerSize; ++inner) {
                T2 index = 0;
                auto maxValue = inputData[outer * axisSize * innerSize + inner];
                for (int32 a = 1; a < axisSize; ++a) {
                    auto curValue = inputData[outer * axisSize * innerSize + a * innerSize + inner];
                    if (curValue > maxValue) {
                        maxValue = curValue;
                        index = a;
                    }
                }
                outputData[outer * innerSize + inner] = index;
            }
        }
    }

    MAI_STATUS run() override {
        const Tensor* inputTensor = getInputTensor(0);
        const Tensor* axisTensor = getInputTensor(1);
        Tensor* outputTensor = getOutputTensor(0);

        MAI_OP_RUN_FIRST_START
        MAI_CHECK_NULL(inputTensor);
        MAI_CHECK_NULL(axisTensor);
        MAI_CHECK_NULL(mParam);
        MAI_CHECK((outputTensor->dataType() == DT_INT32 ||
                outputTensor->dataType() == DT_INT64), "output data type must int32 or int64");
        MAI_CHECK(axisTensor->elementSize() == 1, "Axis must be a single value");
        if (axisTensor->dataType() == DT_INT32) {
            mAxis = *(axisTensor->data<int32>());
        } else if (axisTensor->dataType() == DT_INT64) {
            mAxis = *(axisTensor->data<int64>());
        } else {
            MAI_ABORT("Data type of axis tensor must be int32 or int64");
        }
        int64 dimSize = inputTensor->dimSize();
        MAI_CHECK(mAxis >= -dimSize && mAxis < dimSize,
                "Axis(%d) must be in range [%d, %d)", mAxis, -dimSize, dimSize);
        if (mAxis < 0) {
            mAxis += dimSize;
        }
        MAI_CHECK_NULL(outputTensor);
        std::vector<shape_t> outputShape;
        for (int32 i = 0; i < dimSize; ++i) {
            if (i != mAxis) {
                outputShape.emplace_back(inputTensor->dim(i));
            } else if (mParam->keepDim) {
                outputShape.emplace_back(1);
            }
        }
        if (outputShape.size() == 0) {// output tensor is a scalar
            outputShape.emplace_back(1);
        }
        outputTensor->resize(outputShape);
        // compute mOuterSize & mInnerSize
        for (int32 i = 0; i < dimSize; ++i) {
            if (i < mAxis) {
                mOuterSize *= inputTensor->dim(i);
            } else if (i > mAxis) {
                mInnerSize *= inputTensor->dim(i);
            }
        }
        MAI_OP_RUN_FIRST_END

        int32 axisSize = inputTensor->dim(mAxis);
        if (outputTensor->dataType() == DT_INT32) {
            argMax<T, int32>(inputTensor->data<T>(), outputTensor->mutableData<int32>(),
                    axisSize, mOuterSize, mInnerSize);
        } else if (outputTensor->dataType() == DT_INT64) {
            argMax<T, int64>(inputTensor->data<T>(), outputTensor->mutableData<int64>(),
                    axisSize, mOuterSize, mInnerSize);
        }
        return MAI_SUCCESS;
    }
private:
    int32 mAxis;
    int32 mOuterSize;
    int32 mInnerSize;
    ArgMaxParam* mParam;
    bool mRunFirst;
};

void registerArgMax() {
    MAI_REGISTER_OP((OpContextBuilder().setOperatorType(OP_ARG_MAX).build()), float, ArgMax);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
