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
#include "runtime/GPUDevice.h"

namespace MAI {
namespace Op {
namespace GPU {

template<typename T>
class Relu : public Operator {
public:
    Relu() : mRunFirst(true) {}
    ~Relu() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    MAI_STATUS run() override {
        const Tensor* input = getInputTensor(0);
        Tensor* output = getOutputTensor(0);

        MAI_OP_RUN_FIRST_START
        MAI_CHECK_NULL(input);
        MAI_CHECK_NULL(output);
        output->resize(input->shape());
        MAI_OP_RUN_FIRST_END

        const T* inputData = input->data<T>();
        T* outputData = output->mutableData<T>();
        for (shape_t i = 0; i < input->elementSize(); ++i) {
            outputData[i] = std::max(inputData[i], static_cast<T>(0));
        }
        return MAI_SUCCESS;
    }

    MAI_STATUS run(Context* context) override {
        ALOGI("GPU Relu run context ==================");
        GPUDevice* device = context->device<DEVICE_GPU>();
        //OpenCLRuntime* runtime = device->runtime();
        const Tensor* input = getInputTensor(0);
        Tensor* output = getOutputTensor(0);

        MAI_OP_RUN_FIRST_START
        MAI_CHECK_NULL(input);
        MAI_CHECK_NULL(output);
        output->resize(input->shape());
        MAI_OP_RUN_FIRST_END

        const T* inputData = input->data<T>();
        T* outputData = output->mutableData<T>();
        for (shape_t i = 0; i < input->elementSize(); ++i) {
            outputData[i] = std::max(inputData[i], static_cast<T>(0));
        }
        return MAI_SUCCESS;
    }
private:
    bool mRunFirst;
};

void registerRelu() {
    MAI_REGISTER_OP((OpContextBuilder().setOperatorType(RELU).setDeviceType(DEVICE_GPU).build()), float, Relu);
}

class AA {
public:
    AA() {
        registerRelu();
    }
};

AA __AA;

} // namespace GPU
} // namespace Op
} // namespace MAI
