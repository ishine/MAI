#include "BenchmarkModel.h"

namespace MAI {
namespace Benchmark {

void BenchmarkModel::run() {
    init();
    mListener.onBenchmarkStart();
    run(mCmdParser->get<uint32>("warm_up"), WARM_UP_RUN);
    run(mCmdParser->get<uint32>("num_runs"), NORMAL_RUN);

    mListener.onBenchmarkEnd();
}

void BenchmarkModel::init() {
    mNetwork = std::move(NeuralNetwork::getNeuralNetwork(strToFormat(mCmdParser->get<std::string>("model_format")),
                mCmdParser->get<std::string>("model_path")));
    mNetwork->init();
    mNetwork->setProfiler(&mProfiler);
}

void BenchmarkModel::run(uint32 count, RUN_TYPE runType) {
    for (int i = 0; i < count; ++i) {
        for (int inputIdx = 0; inputIdx < mNetwork->getModelInputs().size(); ++inputIdx) {
            MAI::Tensor* inputTensor = mNetwork->getTensor(mNetwork->getModelInputs()[inputIdx]);
            prepareInput(inputTensor);
        }
        mListener.onSingleRunStart(runType);
        mNetwork->run();
        mListener.onSingleRunEnd();
    }
}

void BenchmarkModel::prepareInput(MAI::Tensor* tensor) {
    if (tensor->dataType() == DT_FLOAT) {
        randomValue<float>(tensor->mutableData<float>(), tensor->elementSize(),
                []() -> float {return static_cast<float>(random()) / RAND_MAX - 0.5f;});
    } else {
        MAI_ABORT("Unsupported now");
    }
}

NeuralNetwork::NetworkFormat BenchmarkModel::strToFormat(const std::string& formatStr) {
    NeuralNetwork::NetworkFormat format = NeuralNetwork::TENSORFLOW;
    if (formatStr == "TENSORFLOW") {
        format = NeuralNetwork::TENSORFLOW;
    } else if (formatStr == "ONNX") {
        format = NeuralNetwork::ONNX;
    } else if (formatStr == "MAI") {
        format = NeuralNetwork::MAI;
    }
    return format;
}

BenchmarkListener::BenchmarkListener(Profiling::Profiler& profiler,
        Profiling::OperatorStatsCalculator& calc) : mProfiler(profiler), mCalc(calc) {
}

void BenchmarkListener::onSingleRunStart(RUN_TYPE runType) {
    if (runType == NORMAL_RUN) {
        mProfiler.setEnable(true);
        mProfiler.reset();
        mProfiler.startProfiling();
    } else if (runType == WARM_UP_RUN) {
        mProfiler.reset();
        mProfiler.setEnable(false);
    }
}

void BenchmarkListener::onSingleRunEnd() {
    mProfiler.stopProfiling();
    auto& events = mProfiler.getProfileEvents();
    mCalc.processSingleRunEvents(events);
}

void BenchmarkListener::onBenchmarkStart() {
}

void BenchmarkListener::onBenchmarkEnd() {
    printf("%s", mCalc.toString().c_str());
}
} // namespace Benchmakr
} // namespace MAI
