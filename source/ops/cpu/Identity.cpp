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

#include <algorithm>
#include "core/OperatorRegister.h"
#include "util/MAIUtil.h"

namespace MAI {
namespace Op {
namespace CPU {

class Identity : public Operator {
public:
    Identity() : mRunFirst(true) {}
    ~Identity() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    MAI_STATUS run() override {
        MAI_OP_RUN_FIRST_START
        const Tensor* inputPtr = getInputTensor(0);
        Tensor* outputPtr = getOutputTensor(0);

        MAI_CHECK_NULL(inputPtr);

        outputPtr->reuse(inputPtr);
        MAI_OP_RUN_FIRST_END

        return MAI_SUCCESS;
    }
private:
    bool mRunFirst;
};

void registerIdentity() {
    MAI_REGISTER_OP((OpContext{.opType=IDENTITY,}), Identity);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
