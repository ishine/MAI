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
#include "runtime/OpenCLUtil.h"
#include "runtime/OpenCLBuffer.h"
#include "tools/profiling/Profiler.h"

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
        return MAI_SUCCESS;
    }

    MAI_STATUS run(Context* context) override {
        ALOGI("GPU Relu run context ==================");
        GPUDevice* device = context->device<DEVICE_GPU>();

        OpenCLRuntime* runtime = reinterpret_cast<OpenCLRuntime*>(device->runtime());
        runtime->buildKernel("Relu", "relu", "", mKernel);
        const Tensor* input = getInputTensor(0);
        Tensor* output = getOutputTensor(0);
        MAI_CHECK_NULL(input);
        MAI_CHECK_NULL(output);
        std::vector<shape_t> shape= input->shape();
        ALOGI("nputshape:%s", shapeToString(shape).c_str());
        output->resize(input->shape());
        ALOGI("after resize");
        mKernel.setArg(0, *reinterpret_cast<OpenCLBuffer*>(input->buffer())->openclBuffer());
        mKernel.setArg(1, *reinterpret_cast<OpenCLBuffer*>(output->buffer())->openclBuffer());
        ALOGI("after setarg");
        uint32 inputSize = static_cast<uint32>(input->elementSize());
        std::vector<uint32> gws = {inputSize};
        //std::vector<uint32> lws = {inputSize};
        std::vector<uint32> lws = {20, 0};
        cl::Event event;
        run1DKernel(runtime, mKernel, gws, lws, event);
        event.wait();
        uint64 start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        uint64 end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        BASIC_OPERATOR_PROFILE(Profiling::Profiler::getInstance(), name(), getNameFromOperator(type()), start / 1000, end / 1000);
        ALOGI("after run cost:%lu ns", (end - start) / 1000);
        return MAI_SUCCESS;
    }
private:
    bool mRunFirst;
    cl::Kernel mKernel;
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
