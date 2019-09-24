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

#include <sys/time.h>
#include <vector>
#include <numeric>

#include "Type.h"
#include "Tensor.h"
#include "util/MAIType.h"

namespace MAI {
    //unit in us
    inline uint64 getCurrentTime() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return static_cast<uint64>(tv.tv_sec) * 1000000 + tv.tv_usec;
    }

    inline shape_t shapeToSizeSkipDim(const std::vector<shape_t>& shape, shape_t dim) {
        if (shape.empty()) {
            return 0;
        }
        shape_t v = 1;
        for (int32 i = 0; i < shape.size(); ++i) {
            if (i != dim) {
                v *= shape[i] ;
            }
        }
        return v;

//        return std::accumulate(shape.begin(), shape.end(), 1,
//                [skipDim = dim](const shape_t& lhs, const shape_t& rhs) -> shape_t {
//                    return lhs * rhs;
//        });
    }

    inline std::string shapeToString(const Tensor* x) {
        std::stringstream ss;
        ss << "[";
        if (x->dimSize() > 0) {
            for (int i = 0; i < x->dimSize() - 1; ++i) {
                ss << x->dim(i) << ", ";
            }
            ss << x->dim(x->dimSize() - 1);
        }
        ss << "]";
        return ss.str();
    }

    inline std::string shapeToStringchenwei(const std::vector<shape_t>& shape) {
        std::stringstream ss;
        ss << "[";
        if (shape.size() > 0) {
            for (int i = 0; i < shape.size() - 1; ++i) {
                ss << shape[i] << ", ";
            }
            ss << shape[shape.size() - 1];
        }
        ss << "]";
        return ss.str();
    }

    template<class T>
    inline std::string shapeToString(const std::vector<T>& shape) {
        std::stringstream ss;
        ss << "[";
        if (shape.size() > 0) {
            for (int i = 0; i < shape.size() - 1; ++i) {
                ss << shape[i] << ", ";
            }
            ss << shape[shape.size() - 1];
        }
        ss << "]";
        return ss.str();
    }

    inline shape_t shapeToSize(const std::vector<shape_t>& shape) {
        if (shape.empty()) {
            return 0;
        }

        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<shape_t>());
    }

    inline int8 getDataTypeSize(DataType dataType) {
        switch(dataType) {
        case DT_FLOAT:
            return sizeof(float);
        case DT_INT32:
            return sizeof(int32);
        case DT_INT64:
            return sizeof(int64);
        default:
            MAI_CHECK(0, "unsupport dataType:%d", dataType);
            break;
        };
    }

    inline shape_t offset4D(const std::vector<shape_t>& shape, shape_t d0, shape_t d1, shape_t d2, shape_t d3) {
        return ((d0 * shape[1] + d1) * shape[2] + d2) * shape[3] + d3;
    }

    template<class T>
    inline bool checkVectorValues(const std::vector<T>& values, T value) {
        for (const T& v : values) {
            if (v != value) {
                return false;
            }
        }
        return true;
    }

    inline std::vector<int32> calcPaddings(PaddingMode paddingMode, const std::vector<int32>& kSize) {
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

    inline std::vector<int32> calculateHW(const std::vector<int32>& hw,
            const std::vector<int32>& kSize,
            const std::vector<int32>& strides,
            const std::vector<int32>& paddings,
            PaddingMode paddingMode) {
        MAI_CHECK(hw.size() == 2, "hw must be 2-d");
        MAI_CHECK(kSize.size() == 2, "kSize must be 2-d");
        MAI_CHECK(strides.size() == 4, "strides must be 4-d");
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
        newHW[0] = calcSize(kSize[0], paddingMode, hw[0], strides[1], paddings.data());
        newHW[1] = calcSize(kSize[1], paddingMode, hw[1], strides[2], paddings.data() + 2);
        return newHW;
    }

} // namespace MAI
