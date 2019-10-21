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

#include <cmath>
#include "ConstFoldOptimizer.h"
#include "NeuralNetwork.h"

namespace MAI {

void ConstFoldOptimizer::optimize() {
    const std::vector<std::string>& opNames = mNeuralNetwork->getOperatorNames();
    int32 opCount = opNames.size();
    std::vector<bool> visited(opCount, false);
    for (int32 i = 0; i < opCount; ++i) {
        Operator* op = mNeuralNetwork->getOperator(opNames[i]);
        dfs(op, visited, i);
    }
    // construct sub graph
    // remove input tensor
    // remove output of op
    // remove op
    printf("SubGraph:%d\n", mUnionFind.subGraphCount());
}

void ConstFoldOptimizer::dfs(Operator* op, std::vector<bool>& visited, int32 index) {
    ALOGI("dfs op:%s, visited:%d, index:%d", op->name().c_str(), visited[index] == false, index);
    if (visited[index]) {
        return;
    }
    ALOGI("start visite:%s", op->name().c_str());

    if (!isComputable(op)) {
        return;
    }

    visited[index] = true;
    for (int32 j = 0; j < op->inputNames().size(); ++j) {
        Tensor* tensor = mNeuralNetwork->getTensor(op->inputName(j));
        mUnionFind.pair({Node::TENSOR, tensor->name()}, {Node::OPERATOR, op->name()});
    }

    for (int32 j = 0; j < op->outputNames().size(); ++j) {
        Tensor* tensor = mNeuralNetwork->getTensor(op->outputName(j));
        tensor->setConst(true);
        mUnionFind.pair({Node::OPERATOR, op->name()}, {Node::TENSOR, tensor->name()});
    }

    for (int32 i = 0; i < mNeuralNetwork->getOperatorNames().size(); ++i) {
        // find edge between op and nextOp
        Operator* nextOp = mNeuralNetwork->getOperator(mNeuralNetwork->getOperatorNames()[i]);
        if (hasEdge(op, nextOp) && isComputable(nextOp)) {
            mUnionFind.pair({Node::OPERATOR, op->name()}, {Node::OPERATOR, nextOp->name()});
            dfs(nextOp, visited, i);
        }
    }
}

bool ConstFoldOptimizer::isComputable(Operator* op) {
    for (int32 j = 0; j < op->inputNames().size(); ++j) {
        Tensor* tensor = mNeuralNetwork->getTensor(op->inputName(j));
        if (!tensor->isConst()) {
            return false;
        }
    }
    return true;
}

bool ConstFoldOptimizer::hasEdge(Operator* op, Operator* nextOp) {
    for (int32 i = 0; i < op->outputNames().size(); ++i) {
        const std::string& curOutputName = op->outputName(i);
        for (int32 j = 0; j < nextOp->inputNames().size(); ++j) {
            const std::string& inputName = nextOp->inputName(j);
            if (curOutputName == inputName) {
                return true;
            }
        }
    }
    return false;
}

} // namespace MAI
