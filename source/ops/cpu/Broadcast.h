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

//static int computeOffset(const std::vector<shape_t>& dims, const std::vector<shape_t>& strides) {
//    int offset = 0;
//    for (int32 i = 0; i < dims.size(); ++i) {
//        offset += dims[i] * strides[i];
//    }
//    ALOGI("computeOffset dims:%s strides:%s offset:%d", shapeToString(dims).c_str(), shapeToString(strides).c_str(), offset);
//    return offset;
//}

template<class TI, class TO, int N, int I = 0>
struct RecursiveFor {
    using BinaryFunc = std::function<void(const TI* inputA, const TI* inputB, TO* output)>;
    static void recursiveFor(
            const TI* inputA, const std::vector<shape_t>& strideA,
            const TI* inputB, const std::vector<shape_t>& strideB,
            TO* output, const std::vector<shape_t>& strideO,
            const std::vector<shape_t>& loopShape,
            BinaryFunc func) {
        for (int i = 0; i < loopShape[I]; ++i) {
            RecursiveFor<TI, TO, N, I + 1>::recursiveFor(
                    inputA + i * strideA[I], strideA,
                    inputB + i * strideB[I], strideB,
                    output + i * strideO[I], strideO, loopShape, func);
        }
    }
};

template<class TI, class TO, int N>
struct RecursiveFor<TI, TO, N, (N - 1)> {
    using BinaryFunc = std::function<void(const TI* inputA, const TI* inputB, TO* output)>;
    static void recursiveFor(
            const TI* inputA, const std::vector<shape_t>& strideA,
            const TI* inputB, const std::vector<shape_t>& strideB,
            TO* output, const std::vector<shape_t>& strideO,
            const std::vector<shape_t>& loopShape,
            BinaryFunc func) {
        constexpr int I = N - 1;
        ALOGI("before last loop>>>>>>>>>>>>>>>");
        for (int i = 0; i < loopShape[I]; ++i) {
            //ALOGI("inputAData offset=%d, bOffset=%d, oOffset=%d\n", strideA[I] * i, strideB[I] * i, strideO[I] * i);
            func(inputA + strideA[I] * i,
                    inputB + strideB[I] * i,
                    output + strideO[I] * i);
            //for (int j = 0; j < dims.size(); j++) {
            //ss << dims[j] << " ";
            //}
            //ss << i;
            //printf("iterator:%s\n", ss.str().c_str());
        }
        ALOGI("after last loop<<<<<<<<<<<<");
    }
};

template<class TI, class TO>
class Broadcast : public Operator {
public:
    using BinaryFunc = std::function<void(const TI* inputA, const TI* inputB, TO* output)>;
    Broadcast() : mRunFirst(true) {}
    ~Broadcast() {
    }

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    virtual MAI_STATUS onCommonCompute(const Tensor* inputA, const Tensor* inputB,
            Tensor* output) = 0;

    virtual MAI_STATUS onScalarCompute(const Tensor* input, const TI inputScalar,
            Tensor* output, bool convertInput = false) = 0;

    virtual MAI_STATUS onBroadcastCompute() = 0;

    MAI_STATUS broadcastCompute(BinaryFunc func) {
        const Tensor* inputA = getInputTensor(0);
        const Tensor* inputB = getInputTensor(1);
        Tensor* output = getOutputTensor(0);
        std::function<void(const TI* inputA, const std::vector<shape_t>& strideA,
            const TI* inputB, const std::vector<shape_t>& strideB,
            TO* output, const std::vector<shape_t>& strideO,
            const std::vector<shape_t>& loopShape,
            BinaryFunc func)> RecursiveForFunc;

        if (output->dimSize() == 4) {
            RecursiveForFunc = RecursiveFor<TI, TO, 4>::recursiveFor;
        } else if (output->dimSize() == 3) {
            RecursiveForFunc = RecursiveFor<TI, TO, 3>::recursiveFor;
        } else if (output->dimSize() == 2) {
            RecursiveForFunc = RecursiveFor<TI, TO, 2>::recursiveFor;
        } else if (output->dimSize() == 1) {
            RecursiveForFunc = RecursiveFor<TI, TO, 1>::recursiveFor;
        } else {
            MAI_ABORT("Unsupport dimSize:%d", output->dimSize());
        }

        RecursiveForFunc(inputA->data<TI>(), mStrideA,
                    inputB->data<TI>(), mStrideB,
                    output->mutableData<TO>(), mStrideO, output->shape(), func);
        return MAI_SUCCESS;
    }


    MAI_STATUS run() override {
        const Tensor* inputA = getInputTensor(0);
        const Tensor* inputB = getInputTensor(1);
        Tensor* output = getOutputTensor(0);

        MAI_OP_RUN_FIRST_START
        MAI_CHECK_NULL(inputA);
        MAI_CHECK_NULL(inputB);
        MAI_CHECK_NULL(output);
        auto outputShape = broadcastShape(inputA->shape(), inputB->shape());
        output->resize(outputShape);
        mStrideA.resize(outputShape.size());
        mStrideB.resize(outputShape.size());
        mStrideO.resize(outputShape.size());
        ALOGI("inputA Shape:%s", shapeToString(inputA->shape()).c_str());
        ALOGI("inputB Shape:%s", shapeToString(inputB->shape()).c_str());
        ALOGI("output Shape:%s", shapeToString(output->shape()).c_str());
        computeStride(inputA, inputB, output);
        ALOGI("strideA:%s", shapeToString(mStrideA).c_str());
        ALOGI("strideB:%s", shapeToString(mStrideB).c_str());
        ALOGI("strideO:%s", shapeToString(mStrideO).c_str());
        MAI_OP_RUN_FIRST_END

        if (isShapeSame(inputA->shape(), inputB->shape())) {
            onCommonCompute(inputA, inputB, output);
        } else if (inputA->elementSize() == 1) {
            onScalarCompute(inputB, *(inputA->data<TI>()), output, true);
        } else if (inputB->elementSize() == 1) {
            onScalarCompute(inputA, *(inputB->data<TI>()), output);
        } else {
            onBroadcastCompute();
        }

        return MAI_SUCCESS;
    }

    void computeStride(const Tensor* inputA, const Tensor* inputB,
            Tensor* output) {
        const auto& outputShape = output->shape();
        shape_t strideA = 1;
        shape_t strideB = 1;
        shape_t strideO = 1;
        int32 dDeltA = outputShape.size() - inputA->dimSize();
        int32 dDeltB = outputShape.size() - inputB->dimSize();
        mStrideA[outputShape.size() - 1] = inputA->dim(inputA->dimSize() - 1) != 1 ? 1 : 0;
        mStrideB[outputShape.size() - 1] = inputB->dim(inputB->dimSize() - 1) != 1 ? 1 : 0;
        mStrideO[outputShape.size() - 1] = 1;
        for (int32 i = outputShape.size() - 2; i >= 0; --i) {
            if (i >= dDeltA) {
                int32 dimA = inputA->dim(i - dDeltA + 1);
                strideA *= dimA;
                mStrideA[i] = inputA->dim(i - dDeltA) != 1 ? strideA : 0;
            } else {
                mStrideA[i] = 0;
            }
            if (i >= dDeltB) {
                int32 dimB = inputB->dim(i - dDeltB + 1);
                strideB *= dimB;
                mStrideB[i] = inputB->dim(i - dDeltB) != 1 ? strideB : 0;
            } else {
                mStrideB[i] = 0;
            }
            strideO *= output->dim(i + 1);
            mStrideO[i] = strideO;
        }
    }
private:
    bool mRunFirst;
    std::vector<shape_t> mStrideA;
    std::vector<shape_t> mStrideB;
    std::vector<shape_t> mStrideO;
};

} // namespace CPU
} // namespace Op
} // namespace MAI
