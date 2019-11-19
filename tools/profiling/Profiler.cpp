#include <sys/time.h>
#include "Profiler.h"
namespace MAI {
namespace Profiling {

Profiler::Profiler() {
    mEventHub = new EventHub();
}

Profiler::~Profiler() {
    if (mEventHub) {
        delete mEventHub;
        mEventHub = NULL;
    }
}

void Profiler::startProfiling() {
    mEventHub->setEnable(true);
}

void Profiler::stopProfiling() {
    mEventHub->setEnable(false);
}

void Profiler::reset() {
    mEventHub->reset();
}

int32_t EventHub::addEvent(const ProfileEvent& event) {
    if (!mEnabled) {
        return -1;
    }
    mEvents.push_back(event);
    return mEvents.size() - 1;
}

int32_t EventHub::beginEvent(const ProfileEvent& event) {
    if (!mEnabled) {
        return -1;
    }
    mEvents.push_back(event);
    mEvents[mEvents.size() - 1].beginTime = nowMicros();
    return mEvents.size() - 1;
}

void EventHub::endEvent(int32_t handle) {
    if (!mEnabled || handle == kInvalidHandle || handle >= mEvents.size()) {
        return;
    }

    mEvents[handle].endTime = nowMicros();
}

uint64_t EventHub::nowMicros() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
}

} // namespace Profiling
} // namespace MAI
