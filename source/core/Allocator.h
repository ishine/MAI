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

#include "util/MAIType.h"
#include "MemoryArena.h"

namespace MAI {

class Allocator {
public:
    virtual ~Allocator() {}
    virtual MemoryInfo allocate(uint64 bytes) = 0;
    virtual void deallocate(MemoryInfo& memInfo) = 0;
};

class CPUAllocator : public Allocator {
public:
    MemoryInfo allocate(uint64 bytes) {
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

    void deallocate(MemoryInfo& memInfo) {
        free(memInfo.ptr);
    }
};

} // namespace MAI
