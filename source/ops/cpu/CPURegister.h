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
DECLARE_REGISTER_OP(CRelu);
DECLARE_REGISTER_OP(LeakyRelu);
DECLARE_REGISTER_OP(Sigmoid);
DECLARE_REGISTER_OP(Softmax);
DECLARE_REGISTER_OP(FusedBatchNorm);
DECLARE_REGISTER_OP(Cast);
DECLARE_REGISTER_OP(Floor);
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
DECLARE_REGISTER_OP(Pack);
DECLARE_REGISTER_OP(Mul);
DECLARE_REGISTER_OP(Pad);
DECLARE_REGISTER_OP(Gemm);
DECLARE_REGISTER_OP(Gather);
DECLARE_REGISTER_OP(Add);
DECLARE_REGISTER_OP(Dropout);
DECLARE_REGISTER_OP(ResizeBilinear);
DECLARE_REGISTER_OP(StridedSlice);
DECLARE_REGISTER_OP(Identity);
DECLARE_REGISTER_OP(TransposeConv2d);
DECLARE_REGISTER_OP(Pow);
DECLARE_REGISTER_OP(Neg);
DECLARE_REGISTER_OP(ArgMin);
DECLARE_REGISTER_OP(ArgMax);
DECLARE_REGISTER_OP(Abs);
DECLARE_REGISTER_OP(Tan);
DECLARE_REGISTER_OP(ATan);
DECLARE_REGISTER_OP(Tanh);
DECLARE_REGISTER_OP(ATanh);
DECLARE_REGISTER_OP(Cos);
DECLARE_REGISTER_OP(ACos);
DECLARE_REGISTER_OP(Cosh);
DECLARE_REGISTER_OP(ACosh);
DECLARE_REGISTER_OP(Sin);
DECLARE_REGISTER_OP(ASin);
DECLARE_REGISTER_OP(Sinh);
DECLARE_REGISTER_OP(ASinh);
DECLARE_REGISTER_OP(Sum);
DECLARE_REGISTER_OP(Square);
DECLARE_REGISTER_OP(Greater);
DECLARE_REGISTER_OP(Less);
DECLARE_REGISTER_OP(Equal);
DECLARE_REGISTER_OP(NotEqual);
DECLARE_REGISTER_OP(LogicalAnd);
DECLARE_REGISTER_OP(LogicalNot);
DECLARE_REGISTER_OP(FloorDiv);
DECLARE_REGISTER_OP(FloorMod);
DECLARE_REGISTER_OP(RealDiv);
DECLARE_REGISTER_OP(All);
DECLARE_REGISTER_OP(Any);

class CPURegister {
public:
    inline static CPURegister* getInstance() {
        static CPURegister cpuRegister;
        return &cpuRegister;
    }
    ~CPURegister() = default;
private:
    CPURegister() {
        REGISTER_OP(Transpose);
        REGISTER_OP(Reshape);
        REGISTER_OP(Shape);
        REGISTER_OP(Squeeze);
        REGISTER_OP(BiasAdd);
        REGISTER_OP(Relu);
        REGISTER_OP(Relu1);
        REGISTER_OP(Relu6);
        REGISTER_OP(CRelu);
        REGISTER_OP(LeakyRelu);
        REGISTER_OP(Sigmoid);
        REGISTER_OP(Softmax);
        REGISTER_OP(FusedBatchNorm);
        REGISTER_OP(Cast);
        REGISTER_OP(Floor);
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
        REGISTER_OP(Pack);
        REGISTER_OP(Mul);
        REGISTER_OP(Pad);
        REGISTER_OP(Gemm);
        REGISTER_OP(Gather);
        REGISTER_OP(Add);
        REGISTER_OP(Dropout);
        REGISTER_OP(ResizeBilinear);
        REGISTER_OP(StridedSlice);
        REGISTER_OP(Identity);
        REGISTER_OP(TransposeConv2d);
        REGISTER_OP(Pow);
        REGISTER_OP(Neg);
        REGISTER_OP(ArgMax);
        REGISTER_OP(ArgMin);
        REGISTER_OP(Abs);
        REGISTER_OP(Tan);
        REGISTER_OP(ATan);
        REGISTER_OP(Tanh);
        REGISTER_OP(ATanh);
        REGISTER_OP(Cos);
        REGISTER_OP(ACos);
        REGISTER_OP(Cosh);
        REGISTER_OP(ACosh);
        REGISTER_OP(Sin);
        REGISTER_OP(ASin);
        REGISTER_OP(Sinh);
        REGISTER_OP(ASinh);
        REGISTER_OP(Sum);
        REGISTER_OP(Square);
        REGISTER_OP(Greater);
        REGISTER_OP(Less);
        REGISTER_OP(Equal);
        REGISTER_OP(NotEqual);
        REGISTER_OP(LogicalAnd);
        REGISTER_OP(LogicalNot);
        REGISTER_OP(FloorMod);
        REGISTER_OP(FloorDiv);
        REGISTER_OP(RealDiv);
        REGISTER_OP(All);
        REGISTER_OP(Any);
    }
};

} // namespace CPU
} // namespace Op
} // namespace MAI
