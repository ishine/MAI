#include "OperatorStatsCalculator.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <queue>

namespace MAI {
namespace Profiling {

void OperatorStatsCalculator::processSingleRunEvents(std::vector<ProfileEvent>& events) {
    if (events.empty()) {
        return;
    }
    std::sort(events.begin(), events.end(),
            [](const ProfileEvent& a, const ProfileEvent& b) {
                return a.beginTime < b.beginTime;
            });
    int32_t nodeNum = 0;
    uint64_t totalTime = 0;
    ProfileEvent& event0 = events[0];
    for (ProfileEvent& event : events) {
        if (mStatInfos.find(event.name) == mStatInfos.end()) {
            mStatInfos.insert({event.name, {}});
            StatInfo& statInfo = mStatInfos[event.name];
            statInfo.name = event.name;
            statInfo.type = event.type;
            statInfo.runOrder = nodeNum++;
        }
        StatInfo& statInfo = mStatInfos[event.name];
        statInfo.startUs.addStat(event.beginTime - event0.beginTime);
        uint64_t executeTime = (event.endTime - event.beginTime);
        totalTime += executeTime;
        statInfo.executionUs.addStat(executeTime);
    }
    mTotalTime.addStat(totalTime);
}

std::ostream& OperatorStatsCalculator::formatField(std::ostream& stream, int width) const {
    stream << "\t" << std::right << std::setw(width) << std::fixed << std::setprecision(3);
    return stream;
}

std::string OperatorStatsCalculator::eventHeaderString(const std::string& title) const {
    std::stringstream ss;
    ss << "=============== " << title << " ===============" << std::endl;
    formatField(ss, 24) << "[type]";
    formatField(ss, 9) << "[start]";
    formatField(ss, 9) << "[first]";
    formatField(ss, 9) << "[avg ms]";
    formatField(ss, 9) << "[min ms]";
    formatField(ss, 9) << "[max ms]";
    formatField(ss, 8) << "[%]";
    ss << "\t" << "[name]";
    return ss.str();
}

std::string OperatorStatsCalculator::eventColString(const StatInfo& statInfo) const {
    const double startMs = statInfo.startUs.avg() / 1000.0;
    const double firstTimeMs = statInfo.executionUs.first() / 1000.0;
    const double avgTimeMs = statInfo.executionUs.avg() / 1000.0;
    const double minTimeMs = statInfo.executionUs.min() / 1000.0;
    const double maxTimeMs = statInfo.executionUs.max() / 1000.0;
    const double percentage = static_cast<double>(statInfo.executionUs.sum()) / mTotalTime.sum() * 100.0;
    std::stringstream ss;
    formatField(ss, 24) << statInfo.type;
    formatField(ss, 9) << startMs;
    formatField(ss, 9) << firstTimeMs;
    formatField(ss, 9) << avgTimeMs;
    formatField(ss, 9) << minTimeMs;
    formatField(ss, 9) << maxTimeMs;
    formatField(ss, 8) << percentage << "%";
    ss << "\t" << statInfo.name;
    return ss.str();
}
std::string OperatorStatsCalculator::getSummary() const {
    std::stringstream ss;
    ss << "Times(ms)" << mTotalTime.toString(1000);
    ss << std::endl;
    return ss.str();
}

std::string OperatorStatsCalculator::toStringByMetric(const std::string& title,
        ProfilingMetric metric, int32_t numLimits) const {
    std::vector<const StatInfo*> stats(mStatInfos.size());
    int32_t i = 0;
    for (auto it = mStatInfos.begin(); it != mStatInfos.end(); ++it) {
        stats[i++] = (&(it->second));
    }
    std::sort(stats.begin(), stats.end(),
            [&metric](const StatInfo* a, const StatInfo* b) {
                if (metric == RUN_ORDER) {
                    return a->runOrder < b->runOrder;
                } else if (metric == TOP_BY_TIME) {// reverse order
                    return a->executionUs.avg() > b->executionUs.avg();
                } else {
                    // ERROR
                    return false;
                }
            });
    std::stringstream ss;
    ss << eventHeaderString(title) << std::endl;
    int32_t count = 0;
    for (const StatInfo* stat : stats) {
        if (numLimits > 0 && ++count > numLimits) {
            break;
        }
        ss << eventColString(*stat) << std::endl;
    }
    ss << std::endl;
    return ss.str();
}

std::string OperatorStatsCalculator::toStringByNodeType() const {
    std::stringstream ss;
    ss << "Number of nodes: " << mStatInfos.size() << std::endl;
    ss << "=============== " << "Summary by node type" << " ===============" << std::endl;
    formatField(ss, 24) << "[Node type]";
    formatField(ss, 10) << "[avg ms]";
    formatField(ss, 11) << "[avg %]";
    formatField(ss, 10) << "[times called]";
    ss << std::endl;

    std::map<std::string, uint64_t> mNodeAvgTimeMs;
    std::map<std::string, uint64_t> mNodeCount;
    uint64_t accumulateAvgTimeMs = 0;
    for (auto it = mStatInfos.begin(); it != mStatInfos.end(); ++it) {
        const StatInfo& stat = it->second;
        if (mNodeAvgTimeMs.find(stat.type) == mNodeAvgTimeMs.end()) {
            mNodeAvgTimeMs[stat.type] = stat.executionUs.avg() / 1000.f;
            mNodeCount[stat.type] = 1;
        } else {
            mNodeAvgTimeMs[stat.type] += stat.executionUs.avg() / 1000.f;
            mNodeCount[stat.type] += 1;
        }
        accumulateAvgTimeMs += stat.executionUs.avg() / 1000.f;
    }
    std::priority_queue<std::pair<uint64_t, std::string>> queueTime;
    for (auto it = mNodeAvgTimeMs.begin(); it != mNodeAvgTimeMs.end(); ++it) {
        queueTime.emplace(it->second, it->first);
    }

    while(!queueTime.empty()) {
        auto entry = queueTime.top();
        queueTime.pop();
        formatField(ss, 24) << entry.second;
        formatField(ss, 10) << entry.first;
        formatField(ss, 11) << entry.first / static_cast<float>(accumulateAvgTimeMs) * 100.f;
        formatField(ss, 10) << mNodeCount[entry.second];
        ss << std::endl;
    }
    return ss.str();

}

std::string OperatorStatsCalculator::toString() const {
    std::stringstream ss;
    ss << toStringByMetric("Run Order", RUN_ORDER, 0);
    ss << toStringByMetric("Top by Computation Time", TOP_BY_TIME, 10);
    ss << toStringByNodeType();
    ss << getSummary();
    return ss.str();
}

} // namespace Profiling
} // namespace MAI
