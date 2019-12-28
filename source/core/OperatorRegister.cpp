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

#include "core/OperatorRegister.h"
#include "util/MAIType.h"

namespace MAI {

OperatorRegister* OperatorRegister::getInstance() {
    static OperatorRegister instance;
    return &instance;
}

void OperatorRegister::registerOperator(const OpContext opContext, const OperatorCreator creator) {
    auto it = mOps.find(opContext);
    MAI_CHECK(it == mOps.end(), "Operator(%s) is already registered", opContext.toString().c_str());
    mOps[opContext] = creator;
}

std::unique_ptr<Operator> OperatorRegister::createOperator(const OpContext& opContext) {
#define COMPARE(value) \
    if (it->first.value != opContext.value) { \
        continue; \
    }

    bool findExactly = false;
    auto find = mOps.end();
    for(auto it = mOps.begin(); it != mOps.end(); ++it) {
        COMPARE(opType)
        COMPARE(deviceType)

        if (it->first.dataType == opContext.dataType
                && it->first.extraInfo == opContext.extraInfo) {
            // find exactly
            find = it;
            break;
        }

        if (find != mOps.end()) {
            continue;
        }

        if (it->first.dataType != DT_INVALID && it->first.dataType != opContext.dataType) {
            continue;
        }

        if (it->first.extraInfo != "" && it->first.extraInfo != opContext.extraInfo) {
            continue;
        }

        // find compatible
        find = it;
    }
#undef COMPARE

    MAI_CHECK(find != mOps.end(), "Operator(%s) is not registered", opContext.toString().c_str());
    auto op = find->second();
    op->setType(opContext.opType);
    return op;
}

} // namespace MAI
