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

#include "core/OperatorTest.h"

namespace MAI {
namespace Test {

class PoolTest : public OperatorTest {
};

template<class T>
void test(MAIOperator op,
        DataType dataType,
        const std::vector<int32>& kernelSizes,
        const std::vector<int32>& strides,
        const PaddingMode& paddingMode,
        const std::vector<shape_t>& inputShape,
        const std::vector<T>& inputData,
        const std::vector<shape_t>& checkShape,
        const std::vector<T>& checkData, DataFormat dataFormat = NHWC) {
    PoolParam* param = new PoolParam();
    param->kernelSizes = kernelSizes;
    param->strides = strides;
    param->paddingMode = paddingMode;
    std::unique_ptr<NeuralNetwork> network = NetworkBuilder()
        .addOperator(OperatorBuilder()
            .setType(op)
            .setDataType(dataType)
            .setInputNames({"input"})
            .setOutputNames({"output"})
            .setParam(param)
            .build())
        .template addTensor<T>("input", inputShape, inputData, dataFormat)
        .template addTensor<T>("output", {}, {})
        .template addTensor<T>("check", checkShape, checkData)
        .build();
    network->init();
    network->run();

    ExpectTensorEQ<T, T>(network->getTensor("output"), network->getTensor("check"));
}

TEST_F(PoolTest, avgPool_floatWithSingleChannelValid_NHWC) {
    test<float>(AVG_POOL, DT_FLOAT, {1,2,2,1}, {1,1,1,1}, PADDING_VALID, {2,2,4,1}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},
            {2,1,3,1}, {3.5f,4.5f,5.5f,11.5f,12.5f,13.5f});
}

TEST_F(PoolTest, avgPool_floatWithSingleChannelSame_NHWC) {
    test<float>(AVG_POOL, DT_FLOAT, {1,2,2,1}, {1,1,1,1}, PADDING_SAME, {2,2,4,1}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},
            {2,2,4,1}, {3.5f,4.5f,5.5f,6.f,5.5f,6.5f,7.5f,8.f,11.5f,12.5f,13.5f,14.f,13.5f,14.5f,15.5f,16.f});
}

TEST_F(PoolTest, avgPool_floatWithMultiChannelValid_NHWC) {
    test<float>(AVG_POOL, DT_FLOAT, {1,2,2,1}, {1,1,1,1}, PADDING_VALID, {1,2,2,2}, {1,2,3,4,5,6,7,8},
            {1,1,1,2}, {4.f,5.f});
}

TEST_F(PoolTest, avgPool_floatWithMultiChannelSame_NHWC) {
    test<float>(AVG_POOL, DT_FLOAT, {1,2,2,1}, {1,1,1,1}, PADDING_SAME, {1,2,2,2}, {1,2,3,4,5,6,7,8},
            {1,2,2,2}, {4.f,5.f,5.f,6.f,6.f,7.f,7.f,8.f});
}

TEST_F(PoolTest, maxPool_floatWithMultiChannelSame_NHWC) {
    test<float>(MAX_POOL, DT_FLOAT, {1,2,2,1}, {1,1,1,1}, PADDING_SAME, {1,2,2,2}, {-1,-2,-3,-4,-5,-6,-7,-8},
            {1,2,2,2}, {-1,-2,-3,-4,-5,-6,-7,-8});
}

TEST_F(PoolTest, globalAveragePoolNCHW) {
    test<float>(GLOBAL_AVG_POOL, DT_FLOAT, {1,1,3,3}/*not used*/, {1,1,1,1}/*not used*/, PADDING_VALID/*not used*/, {1,1,3,3}, {1,2,3,4,5,6,7,8,9},
            {1,1,1,1}, {5}, NCHW);
    test<float>(GLOBAL_AVG_POOL, DT_FLOAT, {1,1,3,3}/*not used*/, {1,1,1,1}/*not used*/, PADDING_VALID/*not used*/, {1,2,3,3},
            {1,2,3,4,5,6,7,8,9,
             10,11,12,13,14,15,16,17,18,
            },
            {1,2,1,1}, {5, 14}, NCHW);
}

} // namespace Test
} // namespace MAI
