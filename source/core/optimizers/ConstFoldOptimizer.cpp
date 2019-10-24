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
#include "source/core/SimpleNeuralNetwork.h"
#include "source/core/OperatorRegister.h"
// remove this
#include "source/util/MAIUtil.h"

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
    //printf("SubGraph:%d\n", mUnionFind.subGraphCount());
    std::map<int32, std::vector<Node>> subGraphs;
    auto& nodes = mUnionFind.nodes();
    auto& nodeIds = mUnionFind.nodeIds();
    for (int32 i = 0; i < nodes.size(); ++i) {
        std::vector<Node>& subGraph = subGraphs[mUnionFind.nodeId(nodes[i])];
        subGraph.emplace_back(nodes[i]);
    }
    for (auto kv : subGraphs) {
        SimpleNeuralNetwork* subNetwork = new SimpleNeuralNetwork();
        auto& vec = kv.second;
        ALOGI(">>>>>>>>>>>>>>>>>>>>>>>>>>>>");
        for (const Node& node : vec) {
            if (node.nodeType == Node::TENSOR) {
                const Tensor* originalTensor = mNeuralNetwork->getTensor(node.name);
                //ALOGI("addTensor:%s", node.name.c_str());
                std::unique_ptr<Tensor> tensor(new Tensor(originalTensor, false));
                subNetwork->addTensor(tensor);
                //ALOGI("addTensor:%s end", node.name.c_str());
            } else if (node.nodeType == Node::OPERATOR) {
                Operator* originalOperator = mNeuralNetwork->getOperator(node.name);
                //ALOGI("addOperator:%s", node.name.c_str());

                //FIXME: got a real data type
                std::unique_ptr<Operator> op =
                    OperatorRegister::getInstance()->createOperator({originalOperator->type(), DT_INT32});
                op->addInputNames(originalOperator->inputNames());
                op->addOutputNames(originalOperator->outputNames());
                op->setName(originalOperator->name());
                op->setType(originalOperator->type());
                //FIXME: This param will be delete by twice
                op->setParam(originalOperator->getParam());
                subNetwork->addOperator(op);
                //ALOGI("addOperator:%s end", node.name.c_str());
            }
            //ALOGI("node:%s", node.name.c_str());
        }
        subNetwork->init();
        subNetwork->run();
        // remove op
        for (const Node& node : vec) {
            if (node.nodeType == Node::OPERATOR) {
                //ALOGI("remove op:%s", node.name.c_str());
                mNeuralNetwork->removeOperator(node.name);
            }
        }
        // remove tensor(out-degree is 0)
        for (const Node& node : vec) {
            if (node.nodeType == Node::TENSOR) {
                int32 degree = mNeuralNetwork->getTensorOutDegree(node.name);
                if (0 == degree) {
                    //ALOGI("remove tensor:%s", node.name.c_str());
                    mNeuralNetwork->removeTensor(node.name);
                } else {
                    Tensor* tensor = mNeuralNetwork->getTensor(node.name);
                    if (!tensor->isConst()) {
                        Tensor* subgraphTensor = subNetwork->getTensor(node.name);
                        const int32* data1 = subgraphTensor->data<int32>();
                        // set the output tensor of the subgraph const.
                        tensor->setConst(true);
                        // copy data;
                        tensor->allocateBuffer(subgraphTensor->shape());
                        tensor->copy(subgraphTensor->data<uint8*>(), subgraphTensor->size());
                    }
                }
            }
        }
        ALOGI("<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    }
}

void ConstFoldOptimizer::dfs(Operator* op, std::vector<bool>& visited, int32 index) {
    //ALOGI("dfs op:%s, visited:%d, index:%d", op->name().c_str(), visited[index] == false, index);
    if (visited[index]) {
        return;
    }
    //ALOGI("start visite:%s", op->name().c_str());

    if (!isComputable(op)) {
        return;
    }

    //ALOGI("isComputable:%s", op->name().c_str());
    visited[index] = true;
    for (int32 j = 0; j < op->inputNames().size(); ++j) {
        Tensor* tensor = mNeuralNetwork->getTensor(op->inputName(j));
        mUnionFind.pair({Node::TENSOR, tensor->name()}, {Node::OPERATOR, op->name()});
    }

    for (int32 j = 0; j < op->outputNames().size(); ++j) {
        Tensor* tensor = mNeuralNetwork->getTensor(op->outputName(j));
        mComputableTensors.emplace_back(tensor->name());
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
        if (!tensor->isConst() && std::find(mComputableTensors.begin(),
                mComputableTensors.end(), tensor->name()) == mComputableTensors.end()) {
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
