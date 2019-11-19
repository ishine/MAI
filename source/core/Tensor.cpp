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

#include "Tensor.h"

#include "core/BufferImpl.h"
#include "core/Allocator.h"
#include "util/MAIUtil.h"
#include "util/MAIType.h"

namespace MAI {

Tensor::Tensor(DataType dataType) :
    mName(""),
    mDataType(dataType),
    mDataFormat(NHWC),
    mBuffer(NULL),
    mAllocator(NULL),
    mIsConst(false),
    mFlag(MEMORY_OWNER | ALLOCATOR_OWNER),
    mN(-1), mH(-1), mW(-1), mC(-1), mI(-1), mO(-1) {
}

Tensor::Tensor(DataType dataType, Allocator* allocator) :
    mName(""),
    mDataType(dataType),
    mDataFormat(NHWC),
    mBuffer(NULL),
    mAllocator(allocator),
    mIsConst(false),
    mFlag(0),
    mN(-1), mH(-1), mW(-1), mC(-1), mI(-1), mO(-1) {
}

Tensor::Tensor(const Tensor* tensor, bool reuseBuffer) {
    mName = tensor->mName;
    mDataType = tensor->mDataType;
    mDataFormat = tensor->mDataFormat;
    mFlag = tensor->mFlag;
    mAllocator = tensor->mAllocator;
    if (reuseBuffer) {
        MAI_CHECK(mBuffer != NULL, "Tensor(%s) cannot reuse buffer as buffer is null", mName.c_str());
        mBuffer = tensor->mBuffer;
        mFlag &= ~MEMORY_OWNER;
    } else {
        mBuffer = NULL;
        if (tensor->mBuffer != NULL) {
            allocateBuffer(tensor->mShape);
            copy(tensor->mBuffer->data(), tensor->size());
        }
        mFlag |= MEMORY_OWNER;
    }
    mIsConst = tensor->mIsConst;
    mFlag &= ~ALLOCATOR_OWNER;
    mN = tensor->mN;
    mH = tensor->mH;
    mW = tensor->mW;
    mC = tensor->mC;
    mI = tensor->mI;
    mO = tensor->mO;
}

Tensor::~Tensor() {
    release();
}

void Tensor::release() {
    if ((mFlag & MEMORY_OWNER) && mBuffer) {
        delete mBuffer;
        mBuffer = NULL;
    }

    if ((mFlag & ALLOCATOR_OWNER) && mAllocator) {
        delete mAllocator;
        mAllocator = NULL;
    }
}

void Tensor::copy(const void* data, int64 len) {
    MAI_CHECK(mBuffer != NULL, "Buffer is null");
    mBuffer->copy((const uint8*)data, 0, len);
}

void Tensor::allocateBuffer(const std::vector<shape_t>& shape) {
    MAI_CHECK(!shape.empty(), "Shape cannot be null");
    MAI_CHECK(mBuffer == NULL && mShape.empty(), "Buffer is not null or shape is not null");
    MAI_CHECK(mAllocator != NULL, "Allocator cannot be null");
    mShape = shape;
    mFlag |= MEMORY_OWNER;
    mBuffer = mAllocator->allocateBuffer(size());
    MAI_CHECK_NULL(mBuffer);
    ALOGI("Tensor::allocateBuffer end");
    //mBuffer = new SimpleBuffer(mAllocator);
    //mBuffer->allocate(size());
}

void Tensor::resize(const std::vector<shape_t>& shape) {
    if (mBuffer != NULL) {
        mShape = shape;
        if (size() > mBuffer->size()) {
            mBuffer->resize(size());
        }
    } else {
        allocateBuffer(shape);
    }
}

void Tensor::zero() {
    MAI_CHECK(mBuffer != NULL,
        "Tensor cannot been reset to zero as buffer is null");
    mBuffer->zero();
}

void Tensor::setConst(bool isConst) {
    mIsConst = isConst;
}

bool Tensor::isConst() const {
    return mIsConst;
}

std::vector<shape_t> Tensor::shape() const {
    return mShape;
}

shape_t Tensor::dim(uint8 i) const {
    return mShape[i];
}

uint8 Tensor::dimSize() const {
    return static_cast<uint8>(mShape.size());
}

uint64 Tensor::elementSize() const {
    return shapeToSize(mShape);
}

uint64 Tensor::size() const {
    uint8 dataSize = getDataTypeSize(mDataType);
    return shapeToSize(mShape) * dataSize;
}

void Tensor::reuse(const Tensor* tensor) {
    mFlag &= ~MEMORY_OWNER;
    mDataType = tensor->mDataType;
    mBuffer = tensor->mBuffer;
    mShape = tensor->mShape;
    mAllocator = tensor->mAllocator;
}

void Tensor::reshape(const std::vector<shape_t>& shape) {
    //TODO (gavinchen) check accumulate size of shape and mShape is equal
    mShape = shape;
}

void Tensor::toFile(const std::string& dir, const std::string& file) {
    std::string filePath = dir;
    if (filePath.at(filePath.size() - 1) != '/') {
        filePath += "/";
    }
    if (file == "") {
        std::string newFile = mName;
        replace(newFile, "/", "_");
        filePath += (newFile + ".txt");
    } else {
        filePath += file;
    }

    FILE* outputFile = fopen(filePath.c_str(), "wb");
    if (outputFile) {
        if (mDataType == DT_FLOAT) {
            fprintf(outputFile, "%s\n", shapeToString(shape()).c_str());
            for(int32 i = 0; i < elementSize(); ++i) {
                fprintf(outputFile, "[%d]: %10.8e\n", i, *(data<float>() + i));
            }
        } else if (mDataType == DT_INT64) {
            for(int32 i = 0; i < elementSize(); ++i) {
                fprintf(outputFile, "[%d]: %d\n", i, *(data<EnumToDataType<DT_INT64>::Type>() + i));
            }
        } else if (mDataType == DT_INT32) {
            for(int32 i = 0; i < elementSize(); ++i) {
                fprintf(outputFile, "[%d]: %d\n", i, *(data<EnumToDataType<DT_INT32>::Type>() + i));
            }
        } else {
            ALOGE("DataType(%s) of Tensor to file is not supported", getNameFromDataType(dataType()).c_str());
        }
        fclose(outputFile);
    } else {
        ALOGE("Cannot open file:%s", filePath.c_str());
    }
}

void Tensor::setDataFormat(DataFormat dataFormat) {
    mDataFormat = dataFormat;
#define __ASSIGN_DATA_FORMAT(DATA_FORMAT)     \
    if (mDataFormat == DATA_FORMAT) {         \
        mN = DataFormatIndex<DATA_FORMAT>::N; \
        mH = DataFormatIndex<DATA_FORMAT>::H; \
        mW = DataFormatIndex<DATA_FORMAT>::W; \
        mC = DataFormatIndex<DATA_FORMAT>::C; \
        mI = DataFormatIndex<DATA_FORMAT>::I; \
        mO = DataFormatIndex<DATA_FORMAT>::O; \
    }
    __ASSIGN_DATA_FORMAT(NHWC);
    __ASSIGN_DATA_FORMAT(NCHW);
    __ASSIGN_DATA_FORMAT(HWIO);
    __ASSIGN_DATA_FORMAT(OHWI);
    __ASSIGN_DATA_FORMAT(HWOI);
    __ASSIGN_DATA_FORMAT(OIHW);
    __ASSIGN_DATA_FORMAT(IOHW);
#undef __ASSIGN_DATA_FORMAT
}

int32 Tensor::n() const {
    MAI_CHECK(mN != -1, "n is -1");
    return mN;
}

int32 Tensor::h() const {
    MAI_CHECK(mH != -1, "h is -1");
    return mH;
}

int32 Tensor::w() const {
    MAI_CHECK(mW != -1, "w is -1");
    return mW;
}

int32 Tensor::c() const {
    MAI_CHECK(mC != -1, "c is -1");
    return mC;
}

int32 Tensor::i() const {
    MAI_CHECK(mI != -1, "i is -1");
    return mI;
}

int32 Tensor::o() const {
    MAI_CHECK(mO != -1, "o is -1");
    return mO;
}

int32 Tensor::dimN() const {
    MAI_CHECK(mN != -1, "n is -1");
    return dim(mN);
}

int32 Tensor::dimH() const {
    MAI_CHECK(mH != -1, "h is -1");
    return dim(mH);
}

int32 Tensor::dimW() const {
    MAI_CHECK(mW != -1, "w is -1");
    return dim(mW);
}

int32 Tensor::dimC() const {
    MAI_CHECK(mC != -1, "c is -1");
    return dim(mC);
}

int32 Tensor::dimI() const {
    MAI_CHECK(mI != -1, "i is -1");
    return dim(mI);
}

int32 Tensor::dimO() const {
    MAI_CHECK(mO != -1, "o is -1");
    return dim(mO);
}
} // namespace MAI
