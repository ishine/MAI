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

#include "Log.h"
#include "BenchmarkModel.h"

namespace MAI {
namespace Benchmark {
   int Main(int argc, char** argv) {
       //CmdParser parser = CmdParser()
       CmdParser* parser = new CmdParser();
       (*parser)
           .add("help", 'h', "Help Info")
           .add<uint32>("num_runs", "Loop run times", false, 50)
           .add<uint32>("warm_up", "warm up run times", false, 1)
           .add<std::string>("model_path", "specified model path", true, "")
           .add<std::string>("model_format", "specified model format", true, "", OneOfReader<std::string>({"TENSORFLOW", "ONNX", "MAI"}));
       parser->parse(argc, argv);
       BenchmarkModel model(parser);
       model.run();
       return 0;
   }
} // namespace Benchmark
} // namespace MAI

int main(int argc, char** argv) {
    return MAI::Benchmark::Main(argc, argv);
}
