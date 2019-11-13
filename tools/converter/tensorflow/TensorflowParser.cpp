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

#include <fcntl.h>
#include <fstream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "source/ops/cpu/CPURegister.h"
#include "TensorflowParser.h"
#include "Type.h"
#include "source/core/Allocator.h"
#include "source/core/OperatorRegister.h"
#include "source/core/OpenMP.h"
#include "source/util/MAIUtil.h"
#include "tools/converter/tensorflow/protos/graph.pb.h"
#include "tools/converter/tensorflow/protos/op_def.pb.h"
#include "tools/profiling/Profiler.h"

namespace MAI {
namespace Converter {
namespace Tensorflow {
std::string& trim(std::string &s)
{
    if (s.empty())
    {
        return s;
    }

    s.erase(0,s.find_first_not_of(" "));
    s.erase(s.find_last_not_of(" ") + 1);
    return s;
}

std::vector<std::string> split(const std::string &s, const std::string &seperator){
    std::vector<std::string> result;
    typedef std::string::size_type string_size;
    string_size i = 0;

    while(i != s.size()){
        int flag = 0;
        while(i != s.size() && flag == 0){
            flag = 1;
            for(string_size x = 0; x < seperator.size(); ++x) {
                if(s[i] == seperator[x]){
                    ++i;
                    flag = 0;
                    break;
                }
            }
        }

        flag = 0;
        string_size j = i;
        while(j != s.size() && flag == 0){
            for(string_size x = 0; x < seperator.size(); ++x) {
                if(s[j] == seperator[x]){
                    flag = 1;
                    break;
                }
            }
            if(flag == 0)
                ++j;
        }
        if(i != j){
            std::string v = s.substr(i, j-i);
            trim(v);
            if (v != "") {
                result.push_back(v);
            }
            i = j;
        }
    }
    return result;
}

class TensorflowParser;
typedef std::function<void(TensorflowParser&, const tensorflow::NodeDef& node)> OpParserFunction;
static std::map<std::string, OpParserFunction> opParsers;

bool registerOpParser(OpParserFunction functionParser, const std::string& names) {
    std::vector<std::string> nameVector = split(names, ",");
    bool r = false;
    for (const std::string& name : nameVector) {
        if (opParsers.find(name) == opParsers.end()) {
            opParsers[name] = functionParser;
        } else {
            r = true;
        }

    }
    return r;
}

class OpParserBase {
public:
    static void processArgType() {
    }

    static std::unique_ptr<Operator> createOperator(MAIOperator opType, DataType dataType) {
        std::unique_ptr<Operator> op = OperatorRegister::getInstance()->createOperator({opType, dataType});
        return op;
    }

    static DataType tf2MIDataType(tensorflow::DataType tfDataType) {
        return (DataType)tfDataType;
    }

    static PaddingMode tf2MIPaddingMode(const std::string& paddingModeStr) {
        if (paddingModeStr == "SAME") {
            return PADDING_SAME;
        }
        if (paddingModeStr == "VALID") {
            return PADDING_VALID;
        }
        return PADDING_INVALID;
    }
};

static void parseAttrs(TensorflowParser& parser, const tensorflow::NodeDef& node, MAIOperator opType, tensorflow::DataType& tfDataType,
        Param* param, std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers) {
    auto& attrs = node.attr();
    for (auto it = attrs.begin(); it != attrs.end(); ++it) {
        auto attrParserIt = attrParsers.find(it->first);
        if (attrParserIt != attrParsers.end()) {
            attrParserIt->second(it->second);
        }
    }
    std::unique_ptr<Operator> op = OpParserBase::createOperator(opType, OpParserBase::tf2MIDataType(tfDataType));
    if (param != NULL) {
        op->setParam(param);
    }
    op->setName(node.name());
    for (int32 i = 0; i < node.input_size(); ++i) {
        op->addInputName(node.input(i));
    }
    op->addOutputName(node.name());
    std::unique_ptr<Tensor> tensor(new Tensor(OpParserBase::tf2MIDataType(tfDataType), new CPUAllocator()));
    tensor->setName(node.name());
    tensor->setDataFormat(NHWC);
    parser.mTFNetwork->addTensor(tensor);
    parser.mTFNetwork->addOperator(op);
}


#define OP_PARSER_NAME(_1, ...) _1

#define OP_PARSER(...) \
    class OP_PARSER_NAME(__VA_ARGS__) : public OpParserBase { \
    private: \
        static void parse(TensorflowParser& parser, const tensorflow::NodeDef& node); \
        static int const result; \
    }; \
    int const OP_PARSER_NAME(__VA_ARGS__)::result = registerOpParser(OP_PARSER_NAME(__VA_ARGS__)::parse, #__VA_ARGS__); \
    void OP_PARSER_NAME(__VA_ARGS__)::parse(TensorflowParser& parser, const tensorflow::NodeDef& node)

#define MULTI_OP_PARSER(...) \

OP_PARSER(Conv2D) {
    Conv2DParam* param = new Conv2DParam();
    param->group = 1;// TF Conv2D cannot support group convolution
    tensorflow::DataType tfDataType;
    std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers = {
        {"strides", [&param](const tensorflow::AttrValue& attr)
            {
                param->strides.insert(param->strides.end(), attr.list().i().begin(), attr.list().i().end());
            }
        },

        {"dilations", [&param](const tensorflow::AttrValue& attr)
            {
                param->dilations.insert(param->dilations.end(), attr.list().i().begin(), attr.list().i().end());
            }
        },

        {"padding", [&param](const tensorflow::AttrValue& attr)
            {
                param->paddingMode = tf2MIPaddingMode(attr.s());
            }
        },

        {"T", [&tfDataType](const tensorflow::AttrValue& attr)
            {
                tfDataType = attr.type();
            }
        },
    };
    parseAttrs(parser, node, CONV2D, tfDataType, param, attrParsers);
}

OP_PARSER(Conv2DBackpropInput) {
    TransposeConv2dParam* param = new TransposeConv2dParam();
    tensorflow::DataType tfDataType;
    std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers = {
        {"strides", [&param](const tensorflow::AttrValue& attr)
            {
                param->strides.insert(param->strides.end(), attr.list().i().begin(), attr.list().i().end());
            }
        },

        {"dilations", [&param](const tensorflow::AttrValue& attr)
            {
                param->dilations.insert(param->dilations.end(), attr.list().i().begin(), attr.list().i().end());
            }
        },

        {"padding", [&param](const tensorflow::AttrValue& attr)
            {
                param->paddingMode = tf2MIPaddingMode(attr.s());
            }
        },

        {"T", [&tfDataType](const tensorflow::AttrValue& attr)
            {
                tfDataType = attr.type();
            }
        },
    };
    parseAttrs(parser, node, TRANSPOSE_CONV2D, tfDataType, param, attrParsers);
    parser.mTFNetwork->getTensor(node.input(1/*filter*/))->setDataFormat(HWOI);
}

OP_PARSER(DepthwiseConv2dNative) {
    DepthwiseConv2dParam* param = new DepthwiseConv2dParam();
    tensorflow::DataType tfDataType;
    std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers = {
        {"strides", [&param](const tensorflow::AttrValue& attr)
            {
                param->strides.insert(param->strides.end(), attr.list().i().begin(), attr.list().i().end());
            }
        },

        {"dilations", [&param](const tensorflow::AttrValue& attr)
            {
                param->dilations.insert(param->dilations.end(), attr.list().i().begin(), attr.list().i().end());
            }
        },

        {"padding", [&param](const tensorflow::AttrValue& attr)
            {
                param->paddingMode = tf2MIPaddingMode(attr.s());
            }
        },

        {"T", [&tfDataType](const tensorflow::AttrValue& attr)
            {
                tfDataType = attr.type();
            }
        },
    };
    parseAttrs(parser, node, DEPTHWISE_CONV2D, tfDataType, param, attrParsers);
}

OP_PARSER(AvgPool) {
    PoolParam* param = new PoolParam();
    tensorflow::DataType tfDataType;
    std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers = {
        {"strides", [&param](const tensorflow::AttrValue& attr)
            {
                param->strides.insert(param->strides.end(), attr.list().i().begin(), attr.list().i().end());
            }
        },

        {"ksize", [&param](const tensorflow::AttrValue& attr)
            {
                param->kernelSizes.insert(param->kernelSizes.end(), attr.list().i().begin(), attr.list().i().end());
            }
        },

        {"padding", [&param](const tensorflow::AttrValue& attr)
            {
                param->paddingMode = tf2MIPaddingMode(attr.s());
            }
        },

        {"T", [&tfDataType](const tensorflow::AttrValue& attr)
            {
                tfDataType = attr.type();
            }
        },
    };
    parseAttrs(parser, node, AVG_POOL, tfDataType, param, attrParsers);
}

OP_PARSER(MaxPool) {
    PoolParam* param = new PoolParam();
    tensorflow::DataType tfDataType;
    std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers = {
        {"strides", [&param](const tensorflow::AttrValue& attr)
            {
                param->strides.insert(param->strides.end(), attr.list().i().begin(), attr.list().i().end());
            }
        },

        {"ksize", [&param](const tensorflow::AttrValue& attr)
            {
                param->kernelSizes.insert(param->kernelSizes.end(), attr.list().i().begin(), attr.list().i().end());
            }
        },

        {"padding", [&param](const tensorflow::AttrValue& attr)
            {
                param->paddingMode = tf2MIPaddingMode(attr.s());
            }
        },

        {"T", [&tfDataType](const tensorflow::AttrValue& attr)
            {
                tfDataType = attr.type();
            }
        },
    };
    parseAttrs(parser, node, MAX_POOL, tfDataType, param, attrParsers);
}

OP_PARSER(FusedBatchNorm) {
    FusedBatchNormParam* param = new FusedBatchNormParam();
    tensorflow::DataType tfDataType;
    std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers = {
        {"epsilon", [&param](const tensorflow::AttrValue& attr)
            {
                param->epsilon = attr.f();
            }
        },

        {"T", [&tfDataType](const tensorflow::AttrValue& attr)
            {
                tfDataType = attr.type();
            }
        },
    };
    parseAttrs(parser, node, FUSED_BATCH_NORM, tfDataType, param, attrParsers);
}

OP_PARSER(Add) {
    tensorflow::DataType tfDataType;
    std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers = {
        {"T", [&tfDataType](const tensorflow::AttrValue& attr)
            {
                tfDataType = attr.type();
            }
        },
    };
    parseAttrs(parser, node, ADD, tfDataType, NULL/*param*/, attrParsers);
}

OP_PARSER(Identity) {
    tensorflow::DataType tfDataType;
    std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers = {
        {"T", [&tfDataType](const tensorflow::AttrValue& attr)
            {
                tfDataType = attr.type();
            }
        },
    };
    parseAttrs(parser, node, IDENTITY, tfDataType, NULL/*param*/, attrParsers);
}

OP_PARSER(Mul) {
    tensorflow::DataType tfDataType;
    std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers = {
        {"T", [&tfDataType](const tensorflow::AttrValue& attr)
            {
                tfDataType = attr.type();
            }
        },
    };
    parseAttrs(parser, node, MUL, tfDataType, NULL/*param*/, attrParsers);
}

OP_PARSER(Sigmoid) {
    tensorflow::DataType tfDataType;
    std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers = {
        {"T", [&tfDataType](const tensorflow::AttrValue& attr)
            {
                tfDataType = attr.type();
            }
        },
    };
    parseAttrs(parser, node, SIGMOID, tfDataType, NULL/*param*/, attrParsers);
}

OP_PARSER(Relu) {
    tensorflow::DataType tfDataType;
    std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers = {
        {"T", [&tfDataType](const tensorflow::AttrValue& attr)
            {
                tfDataType = attr.type();
            }
        },
    };
    parseAttrs(parser, node, RELU, tfDataType, NULL/*param*/, attrParsers);
}

OP_PARSER(Relu6) {
    tensorflow::DataType tfDataType;
    std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers = {
        {"T", [&tfDataType](const tensorflow::AttrValue& attr)
            {
                tfDataType = attr.type();
            }
        },
    };
    parseAttrs(parser, node, RELU6, tfDataType, NULL/*param*/, attrParsers);
}

OP_PARSER(BiasAdd) {
    tensorflow::DataType tfDataType;
    std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers = {
        {"T", [&tfDataType](const tensorflow::AttrValue& attr)
            {
                tfDataType = attr.type();
            }
        },
    };
    parseAttrs(parser, node, BIAS_ADD, tfDataType, NULL/*param*/, attrParsers);
}

OP_PARSER(Squeeze) {
    SqueezeParam* param = new SqueezeParam();
    tensorflow::DataType tfDataType;
    std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers = {
        {"squeeze_dims", [&param](const tensorflow::AttrValue& attr)
            {
                param->squeezeDims.insert(param->squeezeDims.end(), attr.list().i().begin(), attr.list().i().end());
            }
        },

        {"T", [&tfDataType](const tensorflow::AttrValue& attr)
            {
                tfDataType = attr.type();
            }
        },
    };
    parseAttrs(parser, node, SQUEEZE, tfDataType, param, attrParsers);
}

OP_PARSER(Reshape) {
    tensorflow::DataType tfDataType;
    std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers = {
        {"Tshape", [](const tensorflow::AttrValue& attr)
            {
                //TODO: support int32 now
            }
        },

        {"T", [&tfDataType](const tensorflow::AttrValue& attr)
            {
                tfDataType = attr.type();
            }
        },
    };
    parseAttrs(parser, node, RESHAPE, tfDataType, NULL/*param*/, attrParsers);
}

OP_PARSER(Shape) {
    tensorflow::DataType tfDataType;
    std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers = {
        {"out_type", [&tfDataType](const tensorflow::AttrValue& attr)
            {
                //TODO: support int32 now
                tfDataType = attr.type();
            }
        },
    };
    parseAttrs(parser, node, SHAPE, tfDataType, NULL/*param*/, attrParsers);
}

OP_PARSER(Softmax) {
    tensorflow::DataType tfDataType;
    std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers = {
        {"T", [&tfDataType](const tensorflow::AttrValue& attr)
            {
                tfDataType = attr.type();
            }
        },
    };
    parseAttrs(parser, node, SOFTMAX, tfDataType, NULL/*param*/, attrParsers);
}

OP_PARSER(Const) {
    //const tensorflow::OpDef& opDef = parser.mOpList.op(parser.mOpMap[node.op()]);
    const tensorflow::AttrValue& attr = node.attr().at("value");
    const tensorflow::TensorProto& tfTensor = attr.tensor();
    std::unique_ptr<Tensor> tensor(new Tensor((MAI::DataType)tfTensor.dtype(), new CPUAllocator()));
    tensor->setName(node.name());
    int dimSize = tfTensor.tensor_shape().dim_size();
    std::vector<shape_t> dims(dimSize == 0 ? 1 : dimSize);
    dims[0] = dimSize == 0 ? 1 : tfTensor.tensor_shape().dim(0).size();
    for (int i = 1; i < dimSize; ++i) {
        dims[i] = tfTensor.tensor_shape().dim(i).size();
    }
    tensor->setDataFormat(HWIO);// TODO:(gavinchen) setDataFormat should be done in node parser
    tensor->allocateBuffer(dims);
    if (!tfTensor.tensor_content().empty()) {
        tensor->copy(tfTensor.tensor_content().c_str(), tfTensor.tensor_content().size());
    } else {
        DataType dataType = tf2MIDataType(tfTensor.dtype());
        const void* value;
        if (dataType == DT_FLOAT) {
            value = tfTensor.float_val().data();
        } else if (dataType == DT_INT32 || dataType == DT_INT16
                || dataType == DT_INT8 || dataType == DT_UINT8) {
            value = tfTensor.int_val().data();
        } else {
            MAI_ABORT("Unsupported data type:%s", getNameFromDataType(dataType).c_str());
        }
        tensor->copy(value, getSizeFromDataType(dataType));
    }
    tensor->setConst(true);
    parser.mTFNetwork->addTensor(tensor);
}

OP_PARSER(Placeholder) {
    tensorflow::DataType tfDataType;
    std::vector<shape_t> shape;
    std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers = {
        {"dtype", [&tfDataType](const tensorflow::AttrValue& attr)
            {
                tfDataType = attr.type();
            }
        },

        {"shape", [&shape](const tensorflow::AttrValue& attr)
            {
                for (int32 i = 0; i < attr.shape().dim_size(); ++i) {
                    int32 dim = attr.shape().dim(i).size();
                    shape.emplace_back(dim == -1 ? 1 : dim);// TODO:(gavinchen) receive input size from command line
                }
            }
        },
    };
    auto& attrs = node.attr();
    for (auto it = attrs.begin(); it != attrs.end(); ++it) {
        auto attrParserIt = attrParsers.find(it->first);
        if (attrParserIt != attrParsers.end()) {
            attrParserIt->second(it->second);
        }
    }
    parser.mTFNetwork->addModelInput(node.name(), tf2MIDataType(tfDataType), NHWC, shape);
}

OP_PARSER(ResizeBilinear) {
    tensorflow::DataType tfDataType;
    std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers = {
        {"T", [&tfDataType](const tensorflow::AttrValue& attr)
            {
                tfDataType = attr.type();
            }
        },
    };
    parseAttrs(parser, node, RESIZE_BILINEAR, tfDataType, NULL/*param*/, attrParsers);
}

OP_PARSER(Pack) {
    PackParam* param = new PackParam();
    tensorflow::DataType tfDataType;
    std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers = {
        {"T", [&tfDataType](const tensorflow::AttrValue& attr)
            {
                tfDataType = attr.type();
            }
        },
        {"axis", [&param](const tensorflow::AttrValue& attr)
            {
                param->axis = attr.i();
            }
        },
        {"N", [&param](const tensorflow::AttrValue& attr)
            {
                param->num = attr.i();
            }
        },
    };
    parseAttrs(parser, node, PACK, tfDataType, param, attrParsers);
}

OP_PARSER(StridedSlice) {
    StridedSliceParam* param = new StridedSliceParam();
    tensorflow::DataType tfDataType;
    std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers = {
        {"T", [&tfDataType](const tensorflow::AttrValue& attr)
            {
                tfDataType = attr.type();
            }
        },
        {"shrink_axis_mask", [&param](const tensorflow::AttrValue& attr)
            {
                param->shrinkAxisMask = attr.i();
            }
        },
        {"begin_mask", [&param](const tensorflow::AttrValue& attr)
            {
                param->beginMask = attr.i();
            }
        },
        {"end_mask", [&param](const tensorflow::AttrValue& attr)
            {
                param->endMask = attr.i();
            }
        },
        {"ellipsis_mask", [&param](const tensorflow::AttrValue& attr)
            {
                int32 v = attr.i();
                MAI_CHECK(v == 0, "ellipsis_mask(%d) not support now", v);
            }
        },
        {"new_axis_mask", [&param](const tensorflow::AttrValue& attr)
            {
                int32 v = attr.i();
                MAI_CHECK(v == 0, "new_axis_mask(%d) not support now", v);
            }
        },
    };
    parseAttrs(parser, node, STRIDED_SLICE, tfDataType, param, attrParsers);
}

TensorflowParser::TensorflowParser(NeuralNetwork* network) : mTFNetwork(network) {
}

void TensorflowParser::parse(const std::string& netPath) {
    if (!openGraph(netPath)) {
        return;
    }
    ALOGI("op count:%d", mTFGraphDef.node_size());
    //ALOGI("op def count:%d", mOpList.op_size());
    int nodeCount = mTFGraphDef.node_size();
    for (int i = 0; i < nodeCount; ++i) {
        const tensorflow::NodeDef& node = mTFGraphDef.node(i);
        const std::string& opType = node.op();
        //const tensorflow::OpDef& opDef = mOpList.op(mOpMap[opType]);
        if (opParsers.find(opType) != opParsers.end()) {
            opParsers[opType](*this, node);
        } else {
            ALOGE("Unrecognize op:%s", opType.c_str());
        }
    }
}

bool TensorflowParser::openGraph(const std::string& netPath) {
    std::ifstream ifs(netPath, std::ifstream::in | std::ifstream::binary);
    if (!ifs.is_open()) {
        ALOGE("Cannot open graph:%s", netPath.c_str());
        return false;
    }
    google::protobuf::io::IstreamInputStream input(&ifs);
    google::protobuf::io::CodedInputStream codedStream(&input);
    bool success = mTFGraphDef.ParseFromCodedStream(&codedStream);
    if (!success) {
        ALOGE("Cannot parse graph:%s", netPath.c_str());
        return false;
    }

    ifs.close();
    return success;
}

bool TensorflowParser::openOpTxt(const std::string& opDefPath) {
    int fd = open(opDefPath.c_str(), O_RDONLY);
    if (fd < 0) {
        ALOGE("Cannot open opDef:%s", opDefPath.c_str());
        return false;
    }
    google::protobuf::io::FileInputStream input(fd);
    input.SetCloseOnDelete(true);
    if (!google::protobuf::TextFormat::Parse(&input, &mOpList)) {
        ALOGE("Cannot parse opDef:%s", opDefPath.c_str());
        return false;
    }
    for (int i = 0; i < mOpList.op_size(); ++i) {
        const tensorflow::OpDef& op = mOpList.op(i);
        //mOpMap[op.name()] = i;
    }
    return true;
}

} // namespace Tensorflow
} // namespace Converter
} // namespace MAI
