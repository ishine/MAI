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

    inline std::string shapeToString(const std::vector<shape_t>& shape) {
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
        default:
            MAI_CHECK(0, "unsupport dataType:%d", dataType);
            break;
        };
    }

} // namespace MAI
