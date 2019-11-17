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

#pragma once

#include "Allocator.h"

namespace MAI {

class GPUDevice;
class OpenCLAllocator : public Allocator {
public:
    OpenCLAllocator(GPUDevice* device);

    MemoryInfo allocate(uint64 bytes) override;

    void deallocate(MemoryInfo& memInfo) override;

private:
    GPUDevice* mDevice;
};

} // namespace MAI
