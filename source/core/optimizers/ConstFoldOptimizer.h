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

#include "Optimizer.h"
#include "Operator.h"
#include "util/MAIUnionFind.h"

namespace MAI {

struct Node {
    enum NodeType {
        TENSOR,
        OPERATOR,
    };
    NodeType nodeType;
    std::string name;
    //bool operator<(const Node& node) {
    //    if (nodeType != node.nodeType) {
    //        return nodeType < node.nodeType;
    //    }

    //    if (name != node.name) {
    //        return name < node.name;
    //    }
    //}

    bool operator==(const Node& node) {
        return nodeType == node.nodeType && name == node.name;
    }
};

class ConstFoldOptimizer : public Optimizer {
public:
    ConstFoldOptimizer(NeuralNetwork* network) : Optimizer(network) {}
    virtual ~ConstFoldOptimizer() = default;
    void optimize();
private:
    bool hasEdge(Operator* op, Operator* nextOp);
    bool isComputable(Operator* op);
    void dfs(Operator* op, std::vector<bool>& visited, int32 index);

    WeightedUnionFind<Node> mUnionFind;
};

} // namespace MAI
