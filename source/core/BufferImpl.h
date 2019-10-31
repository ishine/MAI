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

#include "Buffer.h"
#include "Allocator.h"

namespace MAI {

class SimpleBuffer : public Buffer {
public:
    SimpleBuffer(Allocator* allocator);

    virtual ~SimpleBuffer();
    virtual MAI_STATUS allocate(int64 bytes);

    virtual void setBufferAddr(const uint8* buffer, uint32 offset = 0);
    virtual void setBufferAddr(uint8* buffer, uint32 offset = 0);
    virtual void copy(const uint8* src, int32 offset, int64 len);

    virtual const uint8* data();
    virtual uint8* mutableData();
    virtual void resize(uint64 len);
    virtual uint64 size();
    virtual void zero();

private:
    Allocator* mAllocator;
    uint32 mOffset;
    uint8* mBufferPtr;
    bool mIsConst;
    uint64 mSize;
    MemoryInfo memoryInfo;
};

} // namespace MAI
