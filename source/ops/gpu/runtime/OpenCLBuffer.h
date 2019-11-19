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

#include "Buffer.h"
#include "OpenCLRuntime.h"

namespace MAI {

class OpenCLAllocator;
class OpenCLBuffer : public Buffer {
public:
    OpenCLBuffer(OpenCLAllocator* allocator);
    virtual ~OpenCLBuffer();
    virtual cl::Buffer* openclBuffer();

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
    void mapBuffer();
private:
    OpenCLAllocator* mAllocator;
    cl::Buffer* mBuffer;
    void* mMappedBuffer;
    int32 mSize;

};

} // namespace MAI
