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

#include "OpenCLBuffer.h"
#include "OpenCLAllocator.h"
#include "util/MAIType.h"

namespace MAI {

OpenCLBuffer::OpenCLBuffer(OpenCLAllocator* allocator)
    : mAllocator(allocator), mBuffer(NULL),
    mMappedBuffer(NULL), mSize(0) {}

OpenCLBuffer::~OpenCLBuffer() {
    if (mMappedBuffer != NULL) {
        mAllocator->unmap(mBuffer, mMappedBuffer);
    }
    MAI_DELETE_PTR(mBuffer);
}

cl::Buffer* OpenCLBuffer::openclBuffer() {
    return mBuffer;
}

MAI_STATUS OpenCLBuffer::allocate(int64 bytes) {
    mSize = bytes;
    MemoryInfo memInfo = mAllocator->allocate(bytes);
    if (memInfo.size != bytes) {
        return MAI_FAILED;
    } else {
        mSize = memInfo.size;
        mBuffer = reinterpret_cast<cl::Buffer*>(memInfo.ptr);
        return MAI_SUCCESS;
    }
}

void OpenCLBuffer::setBufferAddr(const uint8* buffer, uint32 offset) {
    MAI_ABORT("setBufferAddr");
}

void OpenCLBuffer::setBufferAddr(uint8* buffer, uint32 offset) {
    MAI_ABORT("setBufferAddr");
}

void OpenCLBuffer::copy(const uint8* src, int32 offset, int64 len) {
    mapBuffer();
    memcpy(mMappedBuffer, src + offset, len);
}

const uint8* OpenCLBuffer::data() {
    mapBuffer();
    return reinterpret_cast<uint8*>(mMappedBuffer);
}

uint8* OpenCLBuffer::mutableData() {
    MAI_ABORT("mutableData");
    return NULL;
}

void OpenCLBuffer::resize(uint64 len) {
    if (mSize != len) {
        MAI_ABORT("resize(from %d to %d) not support now", mSize, len);
    }
    return;
}

uint64 OpenCLBuffer::size() {
    return mSize;
}

void OpenCLBuffer::zero() {
    MAI_ABORT("zero");
}

void OpenCLBuffer::mapBuffer() {
    MAI_CHECK_NULL(mBuffer);
    if (mMappedBuffer == NULL) {
        mMappedBuffer = mAllocator->mapBuffer(mBuffer, 0, mSize);
    }
}

} // namespace MAI

