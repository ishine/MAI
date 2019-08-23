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

#include <sstream>

namespace MAI {

typedef signed char int8;
typedef signed short int16;
typedef signed int int32;
typedef signed long long int64;
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

typedef uint64 shape_t;

enum MAI_STATUS {
    MAI_SUCCESS,
};

enum MAIOperator {
    RESHAPE = 1,
    SQUEEZE = 2,
};

enum DataType {
    DT_INVALID = 0,
    DT_FLOAT = 1,
    DT_DOUBLE = 2,
    DT_INT32 = 3,
    DT_UINT8 = 4,
    DT_INT16 = 5,
    DT_INT8 = 6,
    DT_STRING = 7,
    DT_COMPLEX64 = 8,
    DT_INT64 = 9,
    DT_BOOL = 10,
    DT_QINT8 = 11,
    DT_QUINT8 = 12,
    DT_QINT32 = 13,
    // DT_BFLOAT16 = 14,
    DT_QINT16 = 15,
    DT_QUINT16 = 16,
    DT_UINT16 = 17,
    DT_HALF = 19,
};

template<class T>
struct DataTypeToEnum;

template<DataType value>
struct EnumToDataType;

#define MAI_MAPPING_DATA_TYPE(DATA_TYPE, ENUM_VALUE)  \
    template<>                                        \
    struct DataTypeToEnum<DATA_TYPE> {                \
        static constexpr DataType value = ENUM_VALUE; \
    };                                                \
    template<>                                        \
    struct EnumToDataType<ENUM_VALUE> {               \
        typedef DATA_TYPE Type;                       \
    };

MAI_MAPPING_DATA_TYPE(float, DT_FLOAT);
MAI_MAPPING_DATA_TYPE(uint8, DT_UINT8);
MAI_MAPPING_DATA_TYPE(int8, DT_INT8);
MAI_MAPPING_DATA_TYPE(int32, DT_INT32);

struct OpContext {
    MAIOperator opType;
    DataType dataType;

    bool operator < (const OpContext& opContext) const {
#define COMPARE(value) \
        if (value != opContext.value) {     \
            return value < opContext.value; \
        }

        COMPARE(opType)
        COMPARE(dataType)

#undef COMPARE
        return false;
    }

    std::string toString() const {
        std::stringstream ss;
        ss << "OpContext {type:"
           << opType
           << ", DataType:"
           << dataType
           << "}";
        return ss.str();
    }
};

} // namespace MAI
