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

#include "include/NeuralNetwork.h"
#include "include/Tensor.h"

namespace MAI {

class Device {
    virtual MAI_STATUS init(const NeuralNetwork& network) = 0;
    virtual MAI_STATUS run(const Tensor& tensor) = 0;
};

} // namespace MAI
