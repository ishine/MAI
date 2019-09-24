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

#include <string>
#include <vector>

#include "Buffer.h"
#include "Type.h"

namespace MAI {

class Allocator;
class Tensor {
public:
    Tensor(DataType dataType);
    Tensor(DataType dataType, Allocator* allocator);
    ~Tensor();


    inline DataType dataType() const {
        return mDataType;
    }

    inline std::string name() const {
        return mName;
    }

    inline void setName(const std::string& name) {
        mName = name;
    }

    inline void setDataFormat(DataFormat dataFormat) {
        mDataFormat = dataFormat;
    }

    inline DataFormat getDataFormat() const {
        return mDataFormat;
    }

    inline bool isScalar() const {
        return 1 == dimSize() && 1 == elementSize();
    }

    virtual void copy(const void* data, int64 len);
    virtual void allocateBuffer(const std::vector<uint64>& shape);
    virtual void resize(const std::vector<uint64>& shape);
    virtual void zero();

    template<typename T>
    inline const T* data() const {
        return reinterpret_cast<const T*>(mBuffer->data());
    }

    template<typename T>
    inline T* mutableData() {
        return reinterpret_cast<T*>(mBuffer->mutableData());
    }

    virtual std::vector<uint64> shape() const;//TODO: (gavinchen) use const std::vector<uint64>&
    virtual uint64 dim(uint8 i) const;
    virtual uint8 dimSize() const;
    virtual uint64 elementSize() const;
    virtual uint64 size() const;
    virtual void reuse(const Tensor* tensor);
    virtual void reshape(const std::vector<shape_t>& tensor);
    virtual void release();
private:
    enum FLAG {
        MEMORY_OWNER = 1 << 0,
        ALLOCATOR_OWNER = 1 << 1,
    };

private:
    std::string mName;
    DataType mDataType;
    DataFormat mDataFormat;
    Buffer* mBuffer;
    std::vector<uint64> mShape;
    Allocator* mAllocator;
    int32 mFlag;
};

} // namespace MAI
