// Copyright 2019 MAI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "NeuralNetwork.h"
#include "tools/profiling/Profiler.h"
#include "tools/profiling/OperatorStatsCalculator.h"

namespace MAI {
namespace Test {

class PerformanceRunner {
public:
    PerformanceRunner(std::unique_ptr<NeuralNetwork>& network);
    virtual ~PerformanceRunner() = default;

    static void setLoopCount(int32 loopCount);
    void run(int32 counts);
    void run();
    void run(Context* context);
    void run(Context* context, int32 counts);

private:
    static int32 sLoopCount;
    std::unique_ptr<NeuralNetwork> mNetwork;
    Profiling::Profiler* mProfiler;
    Profiling::OperatorStatsCalculator mCalc;
};

} // namespace Test
} // namespace MAI
