#pragma once

namespace MAI {

class Allocator {
public:
    virtual uint8* allocate(uint64 bytes) = 0;
    virtual void deallocate(uint8* buffer, uint64 bytes) = 0;
};

class CPUAllocator : public Allocator {
    uint8* allocate(uint64 bytes) {
        if (0 == bytes) {
            return NULL;
        }
        void* data = NULL;
        data = malloc(bytes);
        return (uint8*)data;
    }

    void deallocate(uint8* buffer, uint64 bytes) {
        free(buffer);
    }
};

} // namespace MAI
