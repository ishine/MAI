#pragma once

#include "util/MAIType.h"

namespace MAI {

class Allocator {
public:
    virtual ~Allocator() {}
    virtual uint8* allocate(uint64 bytes) = 0;
    virtual void deallocate(uint8* buffer, uint64 bytes) = 0;
};

class CPUAllocator : public Allocator {
public:
    uint8* allocate(uint64 bytes) {
        if (0 == bytes) {
            return NULL;
        }
        void* data = NULL;
        data = malloc(bytes);
        return (uint8*)data;
    }

    void deallocate(uint8* buffer, uint64 bytes) {
        MAI_UNUSED(bytes);
        free(buffer);
    }
};

} // namespace MAI
