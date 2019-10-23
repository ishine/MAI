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
#include <algorithm>

namespace MAI {

template<typename T>
class WeightedUnionFind {
public:
    WeightedUnionFind() : mSubGraphCount(0) {
    }

    WeightedUnionFind(int n) : mSubGraphCount(n), mIds(n) {
    }

    void addNode(const T& node) {
        if (std::find(mObjects.begin(), mObjects.end(), node) == mObjects.end()) {
            mObjects.emplace_back(node);
            mIds.emplace_back(mIds.size());
            mSubGraphCount++;
        }
    }

    std::vector<T>& nodes() {
        return mObjects;
    }

    std::vector<int>& nodeIds() {
        return mIds;
    }

    int subGraphCount() {
        return mSubGraphCount;
    }

    int nodeId(const T& node) {
        auto it = std::find(mObjects.begin(), mObjects.end(), node);
        if (it == mObjects.end()) {
            return -1;
        }
        int offset = std::distance(mObjects.begin(), it);
        while(mIds[offset] != offset) offset = mIds[offset];
        return mIds[offset];
    }

    void pair(const T& p, const T& q) {
        int pIndex = findIndex(p);
        if (pIndex == mObjects.size()) {
            addNode(p);
        }
        int qIndex = findIndex(q);
        if (qIndex == mObjects.size()) {
            addNode(q);
        }
        int pId = findId(pIndex);
        int qId = findId(qIndex);
        if (pId == qId) return;

        mIds[pId] = qId;

        mSubGraphCount--;
    }

private:
    int findIndex(const T& p) {
        auto it = std::find(mObjects.begin(), mObjects.end(), p);
        auto index = std::distance(mObjects.begin(), it);
        return static_cast<int>(index);
    }

    int findId(int p) {
        int index = 0;
        while(mIds[p] != p) {
            mIds[p] = mIds[mIds[p]];
            p = mIds[p];
            index++;
        }
        return p;
    }

private:
    int mSubGraphCount;
    std::vector<T> mObjects;
    std::vector<int> mIds;

};

} // namespace MAI
