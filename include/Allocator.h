#pragma once

namespace MAI {

class Allocator {
public:
    virtual ~Allocator() {}
    virtual uint8* allocate(uint64 bytes) = 0;
    virtual uint8* allocate(uint64 bytes) = 0;
};

} // namespace MAI
