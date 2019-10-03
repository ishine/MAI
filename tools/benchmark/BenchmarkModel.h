#pragma once

#include "NeuralNetwork.h"
#include "source/util/CmdParser.h"
#include "tools/profiling/Profiler.h"
#include "tools/profiling/OperatorStatsCalculator.h"

namespace MAI {
namespace Benchmark {

enum RUN_TYPE {
    WARM_UP_RUN,
    NORMAL_RUN,
};

class BenchmarkModel;

class BenchmarkListener {
public:
    BenchmarkListener(Profiling::Profiler& profiler, Profiling::OperatorStatsCalculator& calc);
    virtual void onSingleRunStart(RUN_TYPE runType);
    virtual void onSingleRunEnd();
    virtual void onBenchmarkStart();
    virtual void onBenchmarkEnd();
private:
    Profiling::Profiler& mProfiler;
    Profiling::OperatorStatsCalculator& mCalc;
};

class BenchmarkModel {
public:
    BenchmarkModel(CmdParser* cmdParser) : mCmdParser(cmdParser), mListener(mProfiler, mCalc) {
    }

    virtual ~BenchmarkModel() {
        if (mCmdParser) {
            delete mCmdParser;
            mCmdParser = NULL;
        }
    }

    void run();
    void init();
private:
    void run(uint32 count, RUN_TYPE runType);
    void prepareInput(MAI::Tensor* tensor);

    template<class T>
    void randomValue(T* data, shape_t size, const std::function<T()>& randomFunc) {
        for (int i = 0; i < size; ++i) {
            *data++ = randomFunc();
        }
    }
private:
    CmdParser* mCmdParser;
    std::unique_ptr<NeuralNetwork> mNetwork;
    Profiling::Profiler mProfiler;
    Profiling::OperatorStatsCalculator mCalc;
    BenchmarkListener mListener;
};
} // namespace Benchmark
} // namespace MAI
