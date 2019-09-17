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
#include "TensorflowNetwork.h"
#include "Type.h"
#include "source/core/Allocator.h"
#include "graph.pb.h"
#include "op_def.pb.h"

namespace MAI {
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

class TensorflowParser {
public:
    TensorflowParser(TensorflowNetwork* network) : mTFNetwork(network) {
    }

    void parse(const std::string& netPath, const std::string& opDefPath) {
        if (!openGraph(netPath)) {
            return;
        }
        if (!openOpTxt(opDefPath)) {
            return;
        }
        ALOGI("op count:%d", mTFGraphDef.node_size());
        ALOGI("op def count:%d", mOpList.op_size());
        int nodeCount = mTFGraphDef.node_size();
        for (int i = 0; i < nodeCount; ++i) {
            const tensorflow::NodeDef& node = mTFGraphDef.node(i);
            const std::string& opType = node.op();
            if (opType == "Placeholder"){
            } else {
                const tensorflow::OpDef& opDef = mOpList.op(mOpMap[opType]);
                if (opParsers.find(opType) != opParsers.end()) {
                    opParsers[opType](*this, node);
                } else {
                    ALOGE("Unrecognize op:%s", opType.c_str());
                }
            }
        }
    }

private:
    bool openGraph(const std::string& netPath) {
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

    bool openOpTxt(const std::string& opDefPath) {
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
            mOpMap[op.name()] = i;
        }
        return true;
    }
public:
    TensorflowNetwork* mTFNetwork;
    tensorflow::GraphDef mTFGraphDef;
    tensorflow::OpList mOpList;
    std::map<std::string, int> mOpMap;
};

class OpParserBase {
public:
    static void processArgType() {
    }
};


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
    ALOGI("Conv2D");
    auto& attrs = node.attr();
    Conv2DParam* param = new Conv2DParam();
    std::map<std::string, std::function<void(const tensorflow::AttrValue&)>> attrParsers = {
        {"strides", [&param](const tensorflow::AttrValue& attr)
            {
                for(int i = 0; i < attr.list().i_size(); ++i) {
                    param->strides[i] = attr.list().i(i);
                }
            }
        },
    };
    for (auto it = attrs.begin(); it != attrs.end(); ++it) {
        auto attrParserIt = attrParsers.find(it->first);
        if (attrParserIt != attrParsers.end()) {
            attrParserIt->second(it->second);
        }
    }
}

OP_PARSER(Const) {
    const tensorflow::OpDef& opDef = parser.mOpList.op(parser.mOpMap[node.op()]);
    const tensorflow::AttrValue& attr = node.attr().at(opDef.attr(0).name());
    const tensorflow::TensorProto& tfTensor = attr.tensor();
    std::unique_ptr<Tensor> tensor(new Tensor((MAI::DataType)tfTensor.dtype(), new CPUAllocator()));
    tensor->setName(node.name());
    int dimSize = tfTensor.tensor_shape().dim_size();
    std::vector<shape_t> dims(dimSize == 0 ? 1 : dimSize);
    dims[0] = dimSize == 0 ? 1 : tfTensor.tensor_shape().dim(0).size();
    for (int i = 1; i < dimSize; ++i) {
        dims[i] = tfTensor.tensor_shape().dim(i).size();
    }
    tensor->allocateBuffer(dims);
    tensor->copy(tfTensor.tensor_content().c_str(), tfTensor.tensor_content().size());
}

TensorflowNetwork::TensorflowNetwork(const std::string& netPath, const std::string& opDefPath) {
    TensorflowParser parser(this);
    parser.parse(netPath, opDefPath);
}

MAI_STATUS TensorflowNetwork::init() {
    for (auto it = mOperators.begin(); it != mOperators.end(); ++it) {
        (*it)->init();
    }
    return MAI_SUCCESS;
}

MAI_STATUS TensorflowNetwork::run() {
    for (auto it = mOperators.begin(); it != mOperators.end(); ++it) {
        (*it)->run();
    }
    return MAI_SUCCESS;
}

MAI_STATUS TensorflowNetwork::addOperator(std::unique_ptr<Operator>& op) {
    op->setNeuralNetwork(this);
    mOperators.emplace_back(std::move(op));
    return MAI_SUCCESS;
}

MAI_STATUS TensorflowNetwork::addTensor(std::unique_ptr<Tensor>& tensor) {
    //MAI_CHECK(mTensors.find(tensor->name()) == mTensors.end(), "%s has exists", tensor->name().c_str());
    mTensors.emplace(tensor->name(), std::move(tensor));
    return MAI_SUCCESS;
}

Tensor* TensorflowNetwork::getTensor(const std::string& name) {
    return mTensors[name].get();
}

Operator* TensorflowNetwork::getOperator(const std::string& name) {
    for (auto it = mOperators.begin(); it != mOperators.end(); ++it) {
        if ((*it)->name() == name) {
            return it->get();
        }
    }
    return NULL;
}

void TensorflowNetwork::builGraph() {

}

} // namespace MAI
