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

#include <functional>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <vector>
#include <numeric>
#include <stdlib.h>
#include <sstream>

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
        case DT_INT8:
            return sizeof(int8);
        default:
            MAI_CHECK(0, "unsupport dataType:%d", dataType);
            break;
    };
}

inline shape_t offset4D(const std::vector<shape_t>& shape, shape_t d0, shape_t d1, shape_t d2, shape_t d3) {
    return (((d0) * shape[1] + (d1)) * shape[2] + (d2)) * shape[3] + (d3);
}

inline shape_t offset(const std::vector<shape_t>& shape, const std::vector<shape_t>& dims) {
    shape_t stride = 1;
    shape_t offsetValue = 0;
    for (shape_t i = shape.size() - 1; i >= 0; --i) {
        offsetValue += dims[i] * stride;
        stride *= shape[i];
    }
    return offsetValue;
}

inline bool isShapeSame(const std::vector<shape_t>& shape1, const std::vector<shape_t>& shape2) {
    if (shape1.size() != shape2.size()) {
        return false;
    }
    for (int32 i = 0; i < shape1.size(); ++i) {
        if (shape1[i] != shape2[i]) {
            return false;
        }
    }
    return true;
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

template<class T>
inline const T* mapFile(const std::string& path) {
    const char* filePath = path.c_str();
    struct stat fileInfo;
    if (stat(filePath, &fileInfo) >= 0) {
    }
    int modelFileHandle = open(filePath, O_RDONLY);
    const T* modelData = static_cast<const T*>(mmap(nullptr, fileInfo.st_size, PROT_READ,
                MAP_PRIVATE, modelFileHandle, 0));
    return modelData;
}

template<class T>
inline void fillRandomValue(T* ptr, const std::vector<shape_t>& shape, const std::function<T()>& randomFunc) {
    if (shape.size() == 0) {
        return;
    }
    shape_t elementsNum = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<shape_t>());
    for (shape_t i = 0; i < elementsNum; ++i) {
        *ptr++ = randomFunc();
    }
}

inline void fillRandomValue(float* ptr, const std::vector<shape_t>& shape) {
    fillRandomValue<float>(ptr, shape, []() -> float {
            return static_cast<float>(rand()) / RAND_MAX - 0.5f;
            });
}

void replace(std::string& str, const std::string& src, const std::string& dst);

std::vector<int32> calcPaddings(PaddingMode paddingMode, const std::vector<int32>& kSize);

std::vector<int32> calculateHW(const std::vector<int32>& hw,
        const std::vector<int32>& kSize,
        const std::vector<int32>& strides,
        const std::vector<int32>& paddings,
        PaddingMode paddingMode);

uint64_t nowMicros();

bool isShapeCompatible(const std::vector<shape_t>& shape1, const std::vector<shape_t>& shape2);

std::vector<shape_t> broadcastShape(const std::vector<shape_t>& shape1, const std::vector<shape_t>& shape2);

template<typename T>
std::string vectorToString(const std::vector<T>& values, const std::string& sep = ", ") {
    if (values.empty()) {
        return "";
    }
    std::stringstream ss;
    for (int32 i = 0; i < values.size() - 1; ++i) {
        ss << values[i] << sep;
    }
    ss << values[values.size() - 1];
    return ss.str();
}

} // namespace MAI
