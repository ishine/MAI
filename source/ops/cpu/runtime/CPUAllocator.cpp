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

#include "CPUAllocator.h"
#include "CPUDevice.h"
#include "BufferImpl.h"

namespace MAI {

CPUAllocator::CPUAllocator(CPUDevice* device) : mDevice(device) {}

Buffer* CPUAllocator::allocateBuffer(uint64 bytes) {
    Buffer* buffer = new SimpleBuffer(this);
    buffer->allocate(bytes);
    ALOGI("CPUAllocator::allocateBuffer");
    return buffer;
}

MemoryInfo CPUAllocator::allocate(uint64 bytes) {
    MemoryInfo memInfo;
    if (0 == bytes) {
        return memInfo;
    }
    void* data = NULL;
    data = malloc(bytes);
    memInfo.ptr = (uint8*)data;
    memInfo.offset = 0;
    memInfo.size = bytes;
    return memInfo;
}

void CPUAllocator::deallocate(MemoryInfo& memInfo) {
    MAI_DELETE_PTR(memInfo.ptr);
}

} // namespace MAI
