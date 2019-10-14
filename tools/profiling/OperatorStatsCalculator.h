#pragma once
#include <string>
#include <map>
#include <vector>
#include <stdint.h>
#include "Profiler.h"
#include "Stat.h"
namespace MAI {
namespace Profiling {

class OperatorStatsCalculator {
public:
    enum ProfilingMetric {
        RUN_ORDER,
        TOP_BY_TIME,
    };
    void processSingleRunEvents(std::vector<ProfileEvent>& events);

    inline Stat<uint64_t> getTotalTime() const {
        return mTotalTime;
    }
    std::string toString() const;
private:
    struct StatInfo {
        std::string name;
        std::string type;
        int32_t runOrder;
        Stat<uint64_t> startUs;
        Stat<uint64_t> executionUs;
        uint64_t timeCalled;
    };
    std::ostream& formatField(std::ostream& stream, int width) const;

    std::string eventHeaderString(const std::string& title) const;
    std::string eventColString(const StatInfo& statInfo) const;
    std::string getSummary() const;
    std::string toStringByMetric(const std::string& title,
            ProfilingMetric metric, int32_t numLimits) const;
    std::string toStringByNodeType() const;

    Stat<uint64_t> mTotalTime;
    std::map<std::string, StatInfo> mStatInfos;
};

} // namespace Profiling
} // namespace MAI
