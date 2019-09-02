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

template<typename T>
class Squeeze : public Operator {
public:
    Squeeze() = default;
    ~Squeeze() = default;

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    void setParam(Param* param) {
        mParam = reinterpret_cast<SqueezeParam*>(param);
    }

    MAI_STATUS run() override {
        const Tensor* inputPtr = getInputTensor(0);
        Tensor* outputPtr = getOutputTensor(0);
        MAI_CHECK_NULL(inputPtr);
        std::vector<shape_t> needSqueeze(inputPtr->dimSize(), false);
        if (mParam == NULL || mParam->squeezeDims.empty()) {
            for (int32 i = 0; i < inputPtr->dimSize(); ++i) {
                if (1 == inputPtr->dim(i)) {
                    needSqueeze[i] = true;
                }
            }
        } else {
            for (size_t i = 0; i < mParam->squeezeDims.size(); ++i) {
                int32 dim = mParam->squeezeDims[i];
                dim = ((dim >= 0) ? dim : (inputPtr->dimSize() + dim));
                if (inputPtr->dim(dim) != 1) {
                    MAI_ABORT("%d dim of input is not equals to 1", dim);
                } else {
                    needSqueeze[dim] = true;
                }
            }
        }

        std::vector<shape_t> shape;
        for (size_t i = 0; i < needSqueeze.size(); ++i) {
            if (!needSqueeze[i]) {
                shape.emplace_back(inputPtr->dim(i));
            }
        }

        outputPtr->reuse(inputPtr);
        outputPtr->reshape(shape);
        return MAI_SUCCESS;
    }
private:
    SqueezeParam* mParam;
};

void registerSqueeze() {
    MAI_REGISTER_OP((OpContext{.opType=SQUEEZE,}), float, Squeeze);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
