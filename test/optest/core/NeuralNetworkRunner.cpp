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

#include "NetworkBuilder.h"
#include "NeuralNetworkRunner.h"

namespace MAI {
namespace Test {

NeuralNetworkRunner::NeuralNetworkRunner(std::unique_ptr<NeuralNetwork>& network)
    : mNetwork(std::move(network)){}

void NeuralNetworkRunner::run() {
    mNetwork->init();

    mNetwork->run();
}

} // namespace Test
} // namespace MAI
