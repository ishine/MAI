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

#include "CL/cl2.hpp"
#include "OpenCLAllocator.h"
#include "OpenCLBuffer.h"
#include "OpenCLRuntime.h"
#include "GPUDevice.h"
#include "util/MAIType.h"

namespace MAI {

OpenCLAllocator::OpenCLAllocator(GPUDevice* device)
    : mDevice(device),
    mRuntime(reinterpret_cast<OpenCLRuntime*>(mDevice->runtime())) {
}

Buffer* OpenCLAllocator::allocateBuffer(uint64 bytes) {
    OpenCLBuffer* buf = new OpenCLBuffer(this);
    buf->allocate(bytes);
    return buf;
}

MemoryInfo OpenCLAllocator::allocate(uint64 bytes) {
    MemoryInfo memInfo;
    cl_int err;
    cl::Buffer* buf = new cl::Buffer(((OpenCLRuntime*)mDevice->runtime())->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        ALOGE("Allocate gpu buffer failed");
        MAI_DELETE_PTR(buf);
        memInfo.ptr = NULL;
        memInfo.offset = 0;
        memInfo.size = bytes;
    } else {
        memInfo.ptr = reinterpret_cast<uint8*>(buf);
        memInfo.offset = 0;
        memInfo.size = bytes;
    }
    return memInfo;
}

void OpenCLAllocator::deallocate(MemoryInfo& memInfo) {
    MAI_DELETE_PTR(memInfo.ptr);
}

void* OpenCLAllocator::mapBuffer(cl::Buffer* buffer, int32 offset, int32 bytes) {
    auto queue = mRuntime->commandQueue();
    cl_int error;
    void* mappedBuffer = queue.enqueueMapBuffer(*buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
            offset, bytes, nullptr, nullptr, &error);
    if (error != CL_SUCCESS) {
        ALOGE("Map buffer error");
        mappedBuffer = NULL;
    }
    return mappedBuffer;
}

void OpenCLAllocator::unmap(void* buffer, void* mappedBuffer) {
    MAI_CHECK_NULL(buffer);
    MAI_CHECK_NULL(mappedBuffer);
    auto queue = mRuntime->commandQueue();
    cl_int error = queue.enqueueUnmapMemObject(*reinterpret_cast<cl::Buffer*>(buffer), mappedBuffer, NULL, NULL);
    if (error != CL_SUCCESS) {
        ALOGE("unmap error");
    }
}

} // namespace MAI
