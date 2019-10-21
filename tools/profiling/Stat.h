#pragma once
//#include <numeric>
#include <limits>
#include <cmath>
#include <sstream>
namespace MAI {
namespace Profiling {

template<class ValueType>
class Stat {
public:
    Stat() : mFirst(0), mNewest(0),
        mMax(std::numeric_limits<ValueType>::min()),
        mMin(std::numeric_limits<ValueType>::max()),
        mCount(0), mSum(0), mAllSame(true) {
    }

    void addStat(ValueType v) {
        if (mCount == 0) {
            mFirst = v;
        }
        if (v != mFirst) {
            mAllSame = false;
        }
        mNewest = v;
        mMax = std::max(mMax, v);
        mMin = std::min(mMin, v);
        ++mCount;
        mSum += v;
        mAllValues.push_back(v);
    }

    inline void reset() {
        new (this) Stat<ValueType>();
    }

    inline ValueType first() const {
        return mFirst;
    }

    inline ValueType newest() const {
        return mNewest;
    }

    inline ValueType max() const {
        return mMax;
    }

    inline ValueType min() const {
        return mMin;
    }

    inline ValueType sum() const {
        return mSum;
    }

    double avg() const {
        return mCount == 0 ? std::numeric_limits<ValueType>::quiet_NaN()
            : static_cast<double>(mSum) / mCount;
    }

    double stdDev() const {
        if (mAllSame) {
            return 0;
        } else {
            double r = avg();
            double sum = 0;
            for (int i = 0; i < mAllValues.size(); ++i) {
                sum += std::pow((mAllValues[i] - r), 2);
            }
            return std::sqrt(sum / mCount);
        }
    }

    std::string toString(ValueType devided = 1) const {
        std::stringstream ss;
        if (mCount == 0) {
            ss << "count = 0";
        } else if (mAllSame) {
            ss << "count = " << mCount << " curr = " << mNewest / devided;
            if (mCount > 1) {
                ss << "(all same)";
            }
        } else {
            ss << "count = " << mCount << " first " << mFirst / devided
                << " last = " << mNewest / devided << " min = " << mMin / devided
                << " max = " << mMax / devided << " avg = " << avg() / devided
                << " std = " << stdDev();
        }
        return ss.str();
    }
private:
    ValueType mFirst;
    ValueType mNewest;
    ValueType mMax;
    ValueType mMin;
    int64_t mCount;
    ValueType mSum;
    bool mAllSame;
    std::vector<ValueType> mAllValues;
};

}
}
