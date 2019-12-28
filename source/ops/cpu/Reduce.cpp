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

class Reduce : public Operator {
public:
    Reduce() : mParam(NULL), mRunFirst(true) {}
    ~Reduce() {
        if (mParam != NULL) {
            delete mParam;
            mParam = NULL;
        }
    }

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    void setParam(Param* param) override {
        mParam = reinterpret_cast<ReduceParam*>(param);
    }

    Param* getParam() override {
        return mParam;
    }

    virtual void onReduce(const Tensor* input, Tensor* output,
            int32 inputOffset, int32 outputOffset, int32 axisSize, int32 stride) = 0;

    MAI_STATUS run() override {
        const Tensor* input = getInputTensor(0);
        Tensor* output = getOutputTensor(0);

        MAI_OP_RUN_FIRST_START
        MAI_CHECK_NULL(input);
        MAI_CHECK_NULL(output);
        if (mParam == NULL) {
            mParam = new ReduceParam();
            mParam->keepDim = false;
        }
        for (int32 axis : mParam->axes) {
            MAI_CHECK(axis < input->dimSize() && axis >= -input->dimSize(), "Invalid axis:%d", axis);
            mReducedAxis.push_back(axis >= 0 ? axis : axis + input->dimSize());
        }
        // TODO:(gavinchen) from end -> start may be got better performance
        std::sort(mReducedAxis.begin(), mReducedAxis.end());
        std::vector<int32>::iterator iter = std::unique(mReducedAxis.begin(),mReducedAxis.end());
        mReducedAxis.erase(iter,mReducedAxis.end());
        if (mReducedAxis.size() > 1) {
            mTmp1Tensor.reset(new Tensor(input->dataType(), input->allocator()));
            mTmp1Tensor->resize(input->shape());
            mTmp2Tensor.reset(new Tensor(input->dataType(), input->allocator()));
            mTmp2Tensor->resize(input->shape());
        }
        if (!mReducedAxis.empty()) {
            output->resize(input->shape());
        } else {
            output->resize({1});//scalar
        }
        MAI_OP_RUN_FIRST_END

        if (mReducedAxis.empty()) {
            onReduce(input, output, 0, 0, input->elementSize(), 1);
            return MAI_SUCCESS;
        }

        std::vector<shape_t> internShape(input->shape());

        const Tensor* src = input;
        Tensor* dst = NULL;
        auto reduceIndex = [&](int axis) {
            int32 outerSize = 1;
            int32 innerSize = 1;
            for (int32 x = 0; x < axis; ++x) {
                outerSize *= internShape[x];
            }
            for (int32 x = axis + 1; x < internShape.size(); ++x) {
                innerSize *= internShape[x];
            }
            int32 axisSize = input->dim(axis);
            for (int32 o = 0; o < outerSize; ++o) {
                int32 outputOffset = o * innerSize;
                int32 inputOffset = o * innerSize * axisSize;
                for (int32 i = 0; i < innerSize; ++i) {
                    this->onReduce(src, dst, inputOffset + i, outputOffset + i, axisSize, innerSize);
                }
            }
        };

        int32 i = 0;
        for (;i < mReducedAxis.size() - 1; ++i) {
            int32 axis = mReducedAxis[i];
            internShape[axis] = 1;
            src = (i == 0) ? input : (i % 2 == 0 ? mTmp2Tensor.get(): mTmp1Tensor.get());
            dst = (i % 2 == 0 ? mTmp1Tensor.get() : mTmp2Tensor.get());
            reduceIndex(axis);// reduce index step by step
        }
        src = dst == NULL ? input : dst;
        dst = output;

        internShape[mReducedAxis[i]] = 1;
        reduceIndex(mReducedAxis[i]);
        if (mParam->keepDim) {
            output->resize(internShape);
        } else {
            std::vector<shape_t> tmpShape(input->shape());
            std::vector<shape_t> outputShape;
            for (int32 i = 0; i < mReducedAxis.size(); ++i) {
                tmpShape[mReducedAxis[i]] = 0;
            }
            for (int32 i = 0; i < tmpShape.size(); ++i) {
                if (tmpShape[i] != 0) {
                    outputShape.push_back(tmpShape[i]);
                }
            }
            output->resize(outputShape);
        }

        return MAI_SUCCESS;
    }
private:
    ReduceParam* mParam;
    std::vector<int32> mReducedAxis;
    bool mRunFirst;
    std::shared_ptr<Tensor> mTmp1Tensor;
    std::shared_ptr<Tensor> mTmp2Tensor;
};

template<class T>
class Sum : public Reduce {
public:
    void onReduce(const Tensor* input, Tensor* output,
            int32 inputOffset, int32 outputOffset, int32 axisSize, int32 stride) {
        const T* inputData = input->data<T>() + inputOffset;
        T* outputData = output->mutableData<T>() + outputOffset;

        T sum = 0;
        for (int32 a = 0; a < axisSize; ++a) {
            sum += inputData[a * stride];
        }

        *outputData = sum;
    }
};

template<class T>
class All : public Reduce {
public:
    void onReduce(const Tensor* input, Tensor* output,
            int32 inputOffset, int32 outputOffset, int32 axisSize, int32 stride) {
        const T* inputData = input->data<T>() + inputOffset;
        T* outputData = output->mutableData<T>() + outputOffset;

        T result = 1;
        for (int32 a = 0; a < axisSize; ++a) {
            result &= inputData[a * stride];
        }

        *outputData = result;
    }
};

template<class T>
class Any : public Reduce {
public:
    void onReduce(const Tensor* input, Tensor* output,
            int32 inputOffset, int32 outputOffset, int32 axisSize, int32 stride) {
        const T* inputData = input->data<T>() + inputOffset;
        T* outputData = output->mutableData<T>() + outputOffset;

        T result = 0;
        for (int32 a = 0; a < axisSize; ++a) {
            result |= inputData[a * stride];
        }

        *outputData = result;
    }
};

void registerSum() {
    MAI_REGISTER_OP((OpContextBuilder().setOperatorType(SUM).build()), float, Sum);
}

void registerAll() {
    MAI_REGISTER_OP((OpContextBuilder().setOperatorType(ALL).build()), int8, All);
}

void registerAny() {
    MAI_REGISTER_OP((OpContextBuilder().setOperatorType(ANY).build()), int8, Any);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
