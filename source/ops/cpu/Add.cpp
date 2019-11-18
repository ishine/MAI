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

template<typename T>
class Add : public Operator {
public:
    Add() : mRunFirst(true) {}
    ~Add() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    MAI_STATUS run() override {
        //TODO:(gavinchen) support broadcast
        const Tensor* t1 = getInputTensor(0);
        const Tensor* t2 = getInputTensor(1);
        Tensor* output = getOutputTensor(0);
        MAI_OP_RUN_FIRST_START
        MAI_CHECK_NULL(t1);
        MAI_CHECK_NULL(t2);
        MAI_CHECK_NULL(output);
        MAI_CHECK(isShapeSame(t1->shape(), t2->shape()), "t1->shape(%s) != t2->shape(%s)",
                shapeToString(t1->shape()).c_str(), shapeToString(t2->shape()).c_str());
        std::vector<shape_t> outputShape(t1->shape());
        output->resize(outputShape);
        MAI_OP_RUN_FIRST_END
        const T* t1Data = t1->data<T>();
        const T* t2Data = t2->data<T>();
        T* outputData = output->mutableData<T>();
        for (shape_t i = 0; i < t1->elementSize(); ++i) {
            outputData[i] = t1Data[i] + t2Data[i];
        }
        return MAI_SUCCESS;
    }
private:
    bool mRunFirst;
};

void registerAdd() {
    MAI_REGISTER_OP((OpContextBuilder().setOperatorType(ADD).build()), float, Add);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
