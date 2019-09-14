#include <string.h>

#include "BufferImpl.h"
#include "util/MAIType.h"

namespace MAI {

SimpleBuffer::SimpleBuffer(Allocator* allocator) :
    mAllocator(allocator),
    mOffset(0),
    mBufferPtr(NULL),
    mIsConst(false),
    mSize(0) {
}

SimpleBuffer::~SimpleBuffer(){}

MAI_STATUS SimpleBuffer::allocate(int64 bytes) {
    MAI_CHECK(mAllocator != NULL, "Allocator is null");
    MAI_CHECK(mBufferPtr == NULL, "Buffer has been allocated");
    mBufferPtr = mAllocator->allocate(bytes);
    mSize = bytes;
    return MAI_SUCCESS;
}

void SimpleBuffer::setBufferAddr(const uint8* buffer, uint32 offset) {
    MAI_CHECK(mBufferPtr == NULL, "Buffer has been allocated");
    mIsConst = true;
    mBufferPtr = const_cast<uint8*>(buffer);
    mOffset = offset;
}

void SimpleBuffer::setBufferAddr(uint8* buffer, uint32 offset) {
    MAI_CHECK(mBufferPtr == NULL, "Buffer has been allocated");
    mIsConst = false;
    mBufferPtr = buffer;
    mOffset = offset;
}

void SimpleBuffer::copy(const uint8* src, int32 offset, int64 len) {
    MAI_CHECK(mBufferPtr != NULL, "Buffer is null");
    memcpy(mBufferPtr, src + offset, len);
}

const uint8* SimpleBuffer::data() {
    return mBufferPtr;
}

uint8* SimpleBuffer::mutableData() {
    MAI_CHECK(!mIsConst, "Buffer is not mutable");
    return mBufferPtr;
}

void SimpleBuffer::resize(uint64 len) {
    if (mSize != len) {
        delete mBufferPtr;
        mBufferPtr = NULL;
    }
    allocate(len);
}

uint64 SimpleBuffer::size() {
    return mSize;
}

} //namespace MAI
