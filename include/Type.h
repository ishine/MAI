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
#include <map>
#include "Log.h"

namespace MAI {

typedef signed char int8;
typedef signed short int16;
typedef signed int int32;
typedef signed long long int64;
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

typedef int64 shape_t;

enum MAI_STATUS {
    MAI_SUCCESS,
    MAI_FAILED,
};

#define DEFINE_OP_NAME(name) name,
#define DEFINE_OP_NAME_INDEX(name, index) name = index,

#define DEFINE_OP_GET_METHOD(_1, _2, METHOD_NAME) METHOD_NAME

#define DEFINE_OP(...) DEFINE_OP_GET_METHOD( __VA_ARGS__, DEFINE_OP_NAME_INDEX, DEFINE_OP_NAME)(__VA_ARGS__)
enum MAIOperator {
    #include "OperatorType.def"
    //RESHAPE = 1,
    //SQUEEZE = 2,
    //RELU,
    //RELU1,
    //RELU6,
    //SIGMOID,
    //TANH,
    //MAX_POOL,
    //AVG_POOL,
    //BIAS_ADD,
    //FUSED_BATCH_NORM,
    //SHAPE,
    //SOFTMAX,
    //CONV2D,
    //DEPTHWISE_CONV2D,
    //ADD,
    //SUB,
    //DIV,
    //MUL,
    //PACK,
    //UNPACK,
    //PAD,
    //RESIZE_BILINEAR,
    //STRIDED_SLICE,
    //TRANSPOSE_CONV2D,
};

#undef DEFINE_OP_NAME
#undef DEFINE_OP_NAME_INDEX

#define DEFINE_OP_NAME(name) {name, #name},
#define DEFINE_OP_NAME_INDEX(name, index) DEFINE_OP_NAME(name)

inline std::string getNameFromOperator(MAIOperator op) {
    static std::map<MAIOperator, std::string> sOpNameMap = {
        #include "OperatorType.def"
    };

    return sOpNameMap[op];
}

#undef DEFINE_OP_NAME
#undef DEFINE_OP_NAME_INDEX

#define DEFINE_OP_NAME(name) {#name, name},
#define DEFINE_OP_NAME_INDEX(name, index) DEFINE_OP_NAME(name)
inline MAIOperator getOperatorFromName(std::string opName) {
    static std::map<std::string, MAIOperator> sOpNameMap = {
        #include "OperatorType.def"
    };

    return sOpNameMap[opName];
}

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

enum DeviceType {
    DEVICE_CPU,
    DEVICE_GPU,
    DEVICE_DSP,
    DEVICE_NPU,
};

inline std::string getNameFromDataType(DataType dataType) {
    switch(dataType) {
    case DT_INVALID:return "DT_INVALID";
    case DT_FLOAT:return "DT_FLOAT";
    case DT_DOUBLE:return "DT_DOUBLE";
    case DT_INT32:return "DT_INT32";
    case DT_UINT8:return "DT_UINT8";
    case DT_INT16:return "DT_INT16";
    case DT_INT8:return "DT_INT8";
    case DT_STRING:return "DT_STRING";
    case DT_COMPLEX64:return "DT_COMPLEX64";
    case DT_INT64:return "DT_INT64";
    case DT_BOOL:return "DT_BOOL";
    case DT_QINT8:return "DT_QINT8";
    case DT_QUINT8:return "DT_QUINT8";
    case DT_QINT32:return "DT_QINT32";
    case DT_QINT16:return "DT_QINT16";
    case DT_QUINT16:return "DT_QUINT16";
    case DT_UINT16:return "DT_UINT16";
    case DT_HALF:return "DT_HALF";
    default:return "DT_INVALID";
    }
}

inline int32 getSizeFromDataType(DataType dataType) {
    switch(dataType) {
    case DT_INVALID:return -1;
    case DT_FLOAT:return sizeof(float);
    case DT_DOUBLE:return sizeof(double);
    case DT_INT32:return sizeof(int32);
    case DT_UINT8:return sizeof(uint8);
    case DT_INT16:return sizeof(int16);
    case DT_INT8:return sizeof(int8);
    case DT_STRING:return -1;
    case DT_COMPLEX64:return -1;
    case DT_INT64:return sizeof(int64);
    case DT_BOOL:return sizeof(bool);
    case DT_QINT8:return sizeof(int8);
    case DT_QUINT8:return sizeof(uint8);
    case DT_QINT32:return sizeof(int32);
    case DT_QINT16:return sizeof(int16);
    case DT_QUINT16:return sizeof(uint16);
    case DT_UINT16:return sizeof(uint16);
    case DT_HALF:return 16;
    default:return -1;
    }
}

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
MAI_MAPPING_DATA_TYPE(int64, DT_INT64);

enum DataFormat {
    NHWC, // TF conv2d input
    NCHW,
    HWIO, // TF conv2d filter
    OHWI, // ANN conv2d filter
    HWOI, // TF deconv2d filter
    OIHW, // ONNX conv2d filter
    IOHW, // ONNX depthwise conv2d filter
};

inline std::string getNameFromDataFormat(DataFormat format) {
#define DATA_FORMAT_NAME(FORMAT) \
    if (FORMAT == format) { \
        return #FORMAT; \
    }

    DATA_FORMAT_NAME(NHWC);
    DATA_FORMAT_NAME(NCHW);
    DATA_FORMAT_NAME(HWIO);
    DATA_FORMAT_NAME(OHWI);
    DATA_FORMAT_NAME(HWOI);
    DATA_FORMAT_NAME(OIHW);
    DATA_FORMAT_NAME(IOHW);

    ALOGF("Unknown format:%d", format);
    return "";

#undef DATA_FORMAT_NAME
}

template<DataFormat format>
struct DataFormatIndex;

#define DATA_FORMAT_INDEX(format, n, h, w, c, i, o)              \
    template<>                                                   \
    struct DataFormatIndex<format> {                             \
        static const int N = n;                                  \
        static const int H = h;                                  \
        static const int W = w;                                  \
        static const int C = c;                                  \
        static const int I = i;                                  \
        static const int O = o;                                  \
    }

DATA_FORMAT_INDEX(NHWC,0,1,2,3,-1,1);
DATA_FORMAT_INDEX(NCHW,0,2,3,1,-1,1);
DATA_FORMAT_INDEX(HWIO,-1,0,1,-1,2,3);
DATA_FORMAT_INDEX(OHWI,-1,1,2,-1,3,0);
DATA_FORMAT_INDEX(HWOI,-1,0,1,-1,3,2);
DATA_FORMAT_INDEX(OIHW,-1,2,3,-1,1,0);
DATA_FORMAT_INDEX(IOHW,-1,2,3,-1,0,1);

enum PaddingMode {
    PADDING_INVALID,
    PADDING_VALID,
    PADDING_SAME,
    PADDING_FULL,
};

struct OpContext {
    MAIOperator opType;
    DataType dataType;
    DeviceType deviceType;

    bool operator < (const OpContext& opContext) const {
#define COMPARE(value) \
        if (value != opContext.value) {     \
            return value < opContext.value; \
        }

        COMPARE(opType)
        //if (deviceType != opContext.deviceType) {
            ////printf("deviceType:%d target:%d", deviceType, opContext.deviceType);
            //return deviceType < opContext.deviceType;
        //}
        COMPARE(deviceType)
        if (dataType != opContext.dataType) {
            if (dataType == DT_INVALID || opContext.dataType == DT_INVALID) {
                return false;
            }
            return dataType < opContext.dataType;
        }

#undef COMPARE
        return false;
    }

    std::string toString() const {
        std::stringstream ss;
        ss << "OpContext {type:"
           << getNameFromOperator(opType)
           << ", DataType:"
           << getNameFromDataType(dataType)
           << "}";
        return ss.str();
    }
};

class OpContextBuilder {
public:
    OpContextBuilder() {
        mOpContext.opType = INVALID;
        mOpContext.dataType = DT_INVALID;
        mOpContext.deviceType = DEVICE_CPU;
    }

    inline OpContextBuilder& setOperatorType(MAIOperator op) {
        mOpContext.opType = op;
        return *this;
    }

    inline OpContextBuilder& setDataType(DataType dataType) {
        mOpContext.dataType = dataType;
        return *this;
    }

    inline OpContextBuilder& setDeviceType(DeviceType deviceType) {
        mOpContext.deviceType = deviceType;
        return *this;
    }

    inline OpContext build() {
        return mOpContext;
    }
private:
    OpContext mOpContext;
};

#define MAI_DYNAMIC_DIM -1

} // namespace MAI
