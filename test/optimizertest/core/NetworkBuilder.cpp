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
#include <algorithm>

namespace MAI {
namespace Test {

template<>
NetworkBuilder& NetworkBuilder::addRandomTensor<int32>(
        const std::string& name,
        const std::vector<shape_t>& dims,
        const DataFormat dataFormat) {
    shape_t size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<shape_t>());
    std::vector<int32> data(size);
    std::default_random_engine e;
    std::uniform_int_distribution<int32> norm(0, 10);

    std::generate(data.begin(), data.end(),
            [&e, &norm]{
                return norm(e);
            });
    addTensor(name, dims, data, dataFormat);
    return *this;
}

template<>
NetworkBuilder& NetworkBuilder::addRandomTensor<float>(
        const std::string& name,
        const std::vector<shape_t>& dims,
        const DataFormat dataFormat) {
    shape_t size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<shape_t>());
    std::vector<float> data(size);
    std::default_random_engine e;
    std::uniform_real_distribution<float> norm(0, 1);

    std::generate(data.begin(), data.end(),
            [&e, &norm]{
                return norm(e);
            });
    addTensor(name, dims, data, dataFormat);
    return *this;
}

} // namespace Test
} // namespace MAI
