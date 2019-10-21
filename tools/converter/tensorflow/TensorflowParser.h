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
#include <vector>
#include <map>
#include "include/NeuralNetwork.h"
#include "tools/converter/tensorflow/protos/graph.pb.h"
#include "tools/converter/tensorflow/protos/op_def.pb.h"

namespace MAI {
namespace Converter {
namespace Tensorflow {

class TensorflowParser {
public:
    TensorflowParser(NeuralNetwork* network);

    void parse(const std::string& netPath);

private:
    bool openGraph(const std::string& netPath);
    bool openOpTxt(const std::string& opDefPath);

public:
    NeuralNetwork* mTFNetwork;
    tensorflow::GraphDef mTFGraphDef;
    tensorflow::OpList mOpList;
    std::map<std::string, int> mOpMap;
};

} // namespace Tensorflow
} // namespace Converter
} // namespace MAI
