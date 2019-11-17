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

#include "include/Device.h"
#include "OpenCLRuntime.h"
#include "OpenCLAllocator.h"

namespace MAI {

class GPUDevice : public Device {
public:
    GPUDevice();
    Runtime* runtime() override;
    Allocator* allocator() override;
    //OpenCLRuntime* runtime();
    //OpenCLAllocator* allocator();
private:
    OpenCLRuntime mOpenCLRuntime;
    OpenCLAllocator mAllocator;
};

MAI_DECLARE_TYPE_TRAITS(device, DeviceType, DEVICE_GPU, GPUDevice);

} // namespace MAI
