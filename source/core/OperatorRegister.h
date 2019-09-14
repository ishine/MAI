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

#include <map>

#include "Operator.h"

namespace MAI {

class OperatorRegister {
public:
    static OperatorRegister* getInstance() {
        static OperatorRegister instance;
        return &instance;
    }

    typedef std::function<std::unique_ptr<Operator>()> OperatorCreator;

    template<class DerivedType>
    static std::unique_ptr<Operator> opDefaultCreator() {
        return std::unique_ptr<Operator>(new DerivedType());
    }

    void registerOperator(const OpContext opContext, const OperatorCreator creator);

    std::unique_ptr<Operator> createOperator(const OpContext& opContext);

private:
    OperatorRegister() = default;

private:
    std::map<OpContext, OperatorCreator> mOps;
};

#define MAI_REGISTER_OP_WITH_TEMPLATE(OP_CONTEXT, DATA_TYPE, CLASSNAME)       \
do {                                                                          \
    OpContext opContext = OP_CONTEXT;                                         \
    opContext.dataType = DataTypeToEnum<DATA_TYPE>::value;                    \
    OperatorRegister::getInstance()->registerOperator(                        \
        opContext, OperatorRegister::opDefaultCreator<CLASSNAME<DATA_TYPE> >);\
} while(0)

#define MAI_REGISTER_OP_WITH_NO_TEMPLATE(OP_CONTEXT, CLASSNAME)               \
    OperatorRegister::getInstance()->registerOperator(                        \
        OP_CONTEXT, OperatorRegister::opDefaultCreator<CLASSNAME>)

#define MAI_REGISTER_OP_METHOD(_1, _2, _3, NAME, ...) NAME

#define MAI_REGISTER_OP(...)                                                  \
    MAI_REGISTER_OP_METHOD(__VA_ARGS__, MAI_REGISTER_OP_WITH_TEMPLATE,        \
        MAI_REGISTER_OP_WITH_NO_TEMPLATE)(__VA_ARGS__)

} // namespace MAI
