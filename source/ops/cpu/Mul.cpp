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

static shape_t offsetWithBroadcast(const std::vector<shape_t>& shape, const shape_t& n,
        const shape_t& h, const shape_t& w, const shape_t& c) {
    shape_t shapeSize = shape.size();
    shape_t stride = 1;
    shape_t offset = 0;
    if (shapeSize > 0) {
        shape_t dim = (shape[shapeSize - 1] == 1) ? 0 : c;
        offset += dim;
        stride *= shape[shapeSize - 1];
    }

    if (shape.size() > 1) {
        shape_t dim = (shape[shapeSize - 2] == 1) ? 0 : w;
        offset += dim * stride;
        stride *= shape[shapeSize - 2];
    }

    if (shape.size() > 2) {
        shape_t dim = (shape[shapeSize - 3] == 1) ? 0 : h;
        offset += dim * stride;
        stride *= shape[shapeSize - 3];
    }

    if (shape.size() > 3) {
        shape_t dim = (shape[shapeSize - 4] == 1) ? 0 : n;
        offset += dim * stride;
        stride *= shape[shapeSize - 4];
    }

    return offset;
}

template<typename T, int32 dimSize = -1, bool IS_SHAPE_SAME = true>
class MulImpl {
public:
    static void mul(
            const std::vector<shape_t>& input1Shape, const T* input1Data,
            const std::vector<shape_t>& input2Shape, const T* input2Data,
            const std::vector<shape_t>& outputShape, T* outputData) {
        shape_t elementSize =
            std::accumulate(input1Shape.begin(), input1Shape.end(), 1, std::multiplies<shape_t>());
        for (shape_t i = 0; i < elementSize; ++i) {
            outputData[i] = input1Data[i] * input2Data[i];
        }
    }
};


template<typename T>
class MulImpl<T, 4, false> {
public:
    static void mul(
            const std::vector<shape_t>& input1Shape, const T* input1Data,
            const std::vector<shape_t>& input2Shape, const T* input2Data,
            const std::vector<shape_t>& outputShape, T* outputData) {
        for (shape_t n = 0; n < outputShape[0]; ++n) {
            for (shape_t h = 0; h < outputShape[1]; ++h) {
                for (shape_t w = 0; w < outputShape[2]; ++w) {
                    for (shape_t c = 0; c < outputShape[3]; ++c) {
                        *outputData++ = input1Data[offsetWithBroadcast(input1Shape, n, h, w, c)]
                            * input2Data[offsetWithBroadcast(input2Shape, n, h, w, c)];
                    }
                }
            }
        }
    }
};

template<typename T>
class Mul : public Operator {
public:
    Mul() : mRunFirst(true) {}
    ~Mul() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    MAI_STATUS run() override {
        const Tensor* t1 = getInputTensor(0);
        const Tensor* t2 = getInputTensor(1);
        Tensor* output = getOutputTensor(0);

        MAI_OP_RUN_FIRST_START
        MAI_CHECK_NULL(t1);
        MAI_CHECK_NULL(t2);
        MAI_CHECK_NULL(output);
        mShapeSame = isShapeSame(t1->shape(), t2->shape());
        if (mShapeSame) {
            std::vector<shape_t> outputShape(t1->shape());
            output->resize(outputShape);
        } else {
            std::vector<shape_t> outputShape = broadcastShape(t1->shape(), t2->shape());
            output->resize(outputShape);
            if (outputShape.size() != 4) {//TODO:(gavinchen) support more rank
                MAI_ABORT("Unsupport broadcast now for shape not equal to 4");
            }
        }
        MAI_OP_RUN_FIRST_END

        const T* t1Data = t1->data<T>();
        const T* t2Data = t2->data<T>();
        T* outputData = output->mutableData<T>();
        if (mShapeSame) {
            MulImpl<T>::mul(t1->shape(), t1Data, t2->shape(), t2Data, output->shape(), outputData);
        } else {
            MulImpl<T, 4, false>::mul(t1->shape(), t1Data, t2->shape(), t2Data, output->shape(), outputData);
        }
        return MAI_SUCCESS;
    }
private:
    bool mShapeSame;
    bool mRunFirst;
};

void registerMul() {
    MAI_REGISTER_OP((OpContextBuilder().setOperatorType(MUL).build()), float, Mul);
    MAI_REGISTER_OP((OpContextBuilder().setOperatorType(MUL).build()), int32, Mul);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
