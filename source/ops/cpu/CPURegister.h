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

namespace MAI {
namespace Op {
namespace CPU {

#define DECLARE_REGISTER_OP(op) \
    extern void register##op();

#define REGISTER_OP(op) \
    register##op();

DECLARE_REGISTER_OP(Transpose);
DECLARE_REGISTER_OP(Reshape);
DECLARE_REGISTER_OP(Shape);
DECLARE_REGISTER_OP(Squeeze);
DECLARE_REGISTER_OP(BiasAdd);
DECLARE_REGISTER_OP(Relu);
DECLARE_REGISTER_OP(Relu1);
DECLARE_REGISTER_OP(Relu6);
DECLARE_REGISTER_OP(Sigmoid);
DECLARE_REGISTER_OP(Softmax);
DECLARE_REGISTER_OP(FusedBatchNorm);
DECLARE_REGISTER_OP(Cast);
DECLARE_REGISTER_OP(Floor);
DECLARE_REGISTER_OP(Tanh);
DECLARE_REGISTER_OP(Exp);
DECLARE_REGISTER_OP(Fill);
DECLARE_REGISTER_OP(ExpandDims);
DECLARE_REGISTER_OP(Split);
DECLARE_REGISTER_OP(Conv2D);
DECLARE_REGISTER_OP(DepthwiseConv2d);
DECLARE_REGISTER_OP(MaxPool);
DECLARE_REGISTER_OP(AvgPool);
DECLARE_REGISTER_OP(GlobalAvgPool);
DECLARE_REGISTER_OP(Concat);
DECLARE_REGISTER_OP(Mul);
DECLARE_REGISTER_OP(Pad);
DECLARE_REGISTER_OP(Gemm);
DECLARE_REGISTER_OP(Gather);
DECLARE_REGISTER_OP(Add);
DECLARE_REGISTER_OP(Dropout);

class CPURegister {
public:
    CPURegister() {
        REGISTER_OP(Transpose);
        REGISTER_OP(Reshape);
        REGISTER_OP(Shape);
        REGISTER_OP(Squeeze);
        REGISTER_OP(BiasAdd);
        REGISTER_OP(Relu);
        REGISTER_OP(Relu1);
        REGISTER_OP(Relu6);
        REGISTER_OP(Sigmoid);
        REGISTER_OP(Softmax);
        REGISTER_OP(FusedBatchNorm);
        REGISTER_OP(Cast);
        REGISTER_OP(Floor);
        REGISTER_OP(Tanh);
        REGISTER_OP(Exp);
        REGISTER_OP(Fill);
        REGISTER_OP(ExpandDims);
        REGISTER_OP(Split);
        REGISTER_OP(Conv2D);
        REGISTER_OP(DepthwiseConv2d);
        REGISTER_OP(MaxPool);
        REGISTER_OP(AvgPool);
        REGISTER_OP(GlobalAvgPool);
        REGISTER_OP(Concat);
        REGISTER_OP(Mul);
        REGISTER_OP(Pad);
        REGISTER_OP(Gemm);
        REGISTER_OP(Gather);
        REGISTER_OP(Add);
        REGISTER_OP(Dropout);
    }
    ~CPURegister() = default;
};

} // namespace CPU
} // namespace Op
} // namespace MAI
