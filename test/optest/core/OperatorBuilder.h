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

#include "Operator.h"
#include "core/OperatorRegister.h"

namespace MAI {
namespace Test {

class OperatorBuilder {
public:

    OperatorBuilder() : mParam(NULL) {}

    inline OperatorBuilder& setType(MAIOperator opType) {
        mOpContext.opType = opType;
        return *this;
    }

    inline OperatorBuilder& setDataType(DataType dataType) {
        mOpContext.dataType = dataType;
        return *this;
    }

    inline OperatorBuilder& setOpContext(const OpContext& context) {
        mOpContext = context;
        return *this;
    }

    inline OperatorBuilder& setInputNames(const std::vector<std::string>& names) {
        mInputNames.assign(names.begin(), names.end());
        return *this;
    }

    inline OperatorBuilder& setOutputNames(const std::vector<std::string>& names) {
        mOutputNames.assign(names.begin(), names.end());
        return *this;
    }

    inline std::unique_ptr<Operator> build() {
        std::unique_ptr<Operator> op = OperatorRegister::getInstance()->createOperator(mOpContext);
        op->addInputNames(mInputNames);
        op->addOutputNames(mOutputNames);
        if (mParam != NULL) {
            op->setParam(mParam);
        }
        return std::move(op);
    }

    inline OperatorBuilder& setParam(Param* param) {
        mParam = param;
        return *this;
    }

private:
    OpContext mOpContext;
    std::vector<std::string> mInputNames;
    std::vector<std::string> mOutputNames;
    Param* mParam;
};

} // namespace Test
} // namespace MAI
