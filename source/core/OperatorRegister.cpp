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
    auto it = mOps.find(opContext);
    MAI_CHECK(it != mOps.end(), "Operator(%s) is not registered", opContext.toString().c_str());
    auto op = it->second();
    op->setType(opContext.opType);
    return op;
}

} // namespace MAI
