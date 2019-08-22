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

#include "Type.h"

namespace MAI {

class Allocator;
class Buffer {
public:
    virtual ~Buffer() {}
    virtual MAI_STATUS allocate(int64 bytes) = 0;

    virtual void setBufferAddr(const uint8* buffer, uint32 offset = 0) = 0;
    virtual void setBufferAddr(uint8* buffer, uint32 offset = 0) = 0;
    virtual void copy(const uint8* src, int32 offset, int64 len) = 0;

    virtual const uint8* data() = 0;
    virtual uint8* mutableData() = 0;
    virtual void resize(uint64 len) = 0;
    virtual uint64 size() = 0;

    //template<typename T = uint8>
    //const T* data() {
    //    return reinterpret_cast<const T*>(mBufferPtr + mOffset);
    //}

    //template<typename T = uint8>
    //T* mutableData() {
    //    return reinterpret_cast<T*>(mBufferPtr + mOffset);
    //}

};

} // namespace MAI
