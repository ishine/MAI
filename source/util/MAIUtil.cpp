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

#include "util/MAIUtil.h"
#if defined(_MSC_VER)
#include <chrono>
#else
#include <sys/time.h>
#endif

namespace MAI {

std::vector<int32> calcPaddings(PaddingMode paddingMode, const std::vector<int32>& kSize) {
    std::vector<int32> paddings(4);
    int32 paddingHTSize = 0;
    int32 paddingHBSize = 0;
    int32 paddingWLSize = 0;
    int32 paddingWRSize = 0;
    switch (paddingMode) {
        case SAME:
            if (kSize[0] % 2 != 0) {
                paddingHTSize = (kSize[0] - 1) / 2;
                paddingHBSize = (kSize[0] - 1) / 2 + 1;
                paddingWLSize = (kSize[1] - 1) / 2;
                paddingWRSize = (kSize[1] - 1) / 2 + 1;
            } else {
                paddingHTSize = (kSize[0] - 1) / 2;
                paddingHBSize = (kSize[0] - 1) / 2;
                paddingWLSize = (kSize[1] - 1) / 2;
                paddingWRSize = (kSize[1] - 1) / 2;
            }
            break;
        case FULL:
            paddingHTSize = paddingHBSize = (kSize[0] - 1);
            paddingWLSize = paddingWRSize = (kSize[1] - 1);
            break;
        default:
            break;
    };
    paddings[0] = paddingHTSize;
    paddings[1] = paddingHBSize;
    paddings[2] = paddingWLSize;
    paddings[3] = paddingWRSize;
    return paddings;
}

std::vector<int32> calculateHW(const std::vector<int32>& hw,
        const std::vector<int32>& kSize,
        const std::vector<int32>& strides,
        const std::vector<int32>& paddings,
        PaddingMode paddingMode) {
    MAI_CHECK(hw.size() == 2, "hw must be 2-d");
    MAI_CHECK(kSize.size() == 2, "kSize must be 2-d");
    MAI_CHECK(strides.size() == 2, "strides must be 2-d");
    MAI_CHECK(paddingMode != INVALID || paddings.size() == 4, "paddings must be 4-d, but not:%d", paddings.size());
    auto calcSize = [](int32 k, PaddingMode paddingMode, int32 iSize, int32 s, const int32* paddings) -> int32 {
        int32 p = 0;
        switch(paddingMode) {
            case SAME:
                p = (k - 1);
                break;
            case FULL:
                p = (k - 1) * 2;
                break;
            case VALID:
                p = 0;
                break;
            case INVALID:
                p = paddings[0] + paddings[1];
        };
        int32 size = (iSize - k + p) / s + 1;
        return size;
    };
    std::vector<int32> newHW(2);
    newHW[0] = calcSize(kSize[0], paddingMode, hw[0], strides[0], paddings.data());
    newHW[1] = calcSize(kSize[1], paddingMode, hw[1], strides[1], paddings.data() + 2);
    return newHW;
}

void replace(std::string& str, const std::string& src, const std::string& dst) {
    std::string::size_type pos = str.find(src);
    while(pos != std::string::npos) {
        str.replace(pos, src.size(), dst);
        pos = str.find(src);
    }
}

bool isShapeCompatible(const std::vector<shape_t>& shape1, const std::vector<shape_t>& shape2) {
    const std::vector<shape_t>& larger = shape1.size() >= shape2.size() ? shape1 : shape2;
    const std::vector<shape_t>& smaller = shape1.size() >= shape2.size() ? shape2 : shape1;
    for (shape_t i = larger.size() - 1; i >= 0; --i) {
        if (larger.size() - i > smaller.size()) {
            return true;
        }
        if (larger[i] != smaller[i] || larger[i] != 1 || smaller[i] != 1) {
            return false;
        }
    }
    return true;
}

std::vector<shape_t> broadcastShape(
        const std::vector<shape_t>& shape1, const std::vector<shape_t>& shape2) {
    std::vector<shape_t> outputShape(shape1.size() > shape2.size() ? shape1.size() : shape2.size());
    for (int32 i = static_cast<int32>(outputShape.size() - 1); i >= 0; --i) {
        if ((outputShape.size() - i) > shape1.size()) {
            outputShape[i] = shape2[i];
        } else if ((outputShape.size() - i) > shape2.size()) {
            outputShape[i] = shape1[i];
        } else {
            outputShape[i] = shape1[i] == 1 ? shape2[i] : shape1[i];
        }
    }
    return outputShape;
}

#if defined(_MSC_VER)
//time in us
uint64_t nowMicros() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
        .count();
}
#else
//time in us
uint64_t nowMicros() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
}
#endif

} // namespace MAI
