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
#include <vector>
#include <stack>
#include <functional>

namespace MAI {

template<typename T>
void dfs(const std::vector<T>& srcNodes, std::stack<T>& dstNodes,
        typename std::vector<T>::size_type index,
        std::vector<bool>& visited,
        std::function<bool(const T&, const T&)> hasEdge) {
    if (!visited[index]) {
        visited[index] = true;
        for(typename std::vector<T>::size_type i = 0; i < srcNodes.size(); ++i) {
            if(hasEdge(srcNodes[index], srcNodes[i])) {
                dfs(srcNodes, dstNodes, i, visited, hasEdge);
            }
        }
        dstNodes.push(srcNodes[index]);
    }
}

template<typename T>
void topologicalSort(const std::vector<T>& srcNodes, std::vector<T>& dstNodes,
        std::function<bool(const T&, const T&)> hasEdge) {
    std::vector<bool> visited(srcNodes.size(), false);
    std::stack<T> nodes;
    for (typename std::vector<T>::size_type i = 0; i < srcNodes.size(); ++i) {
        if (!visited[i]) {
            dfs(srcNodes, nodes, i, visited, hasEdge);
        }
    }

    while(!nodes.empty()) {
        dstNodes.push_back(nodes.top());
        nodes.pop();
    }
}

} // namespace MAI
