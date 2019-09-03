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
namespace Test {

inline bool isDimsSame(const Tensor* x, const Tensor* y) {
    return x->shape() == y->shape();
}

inline void ExpectDimsEQ(const Tensor* x, const Tensor* y) {
    ASSERT_TRUE(x->shape() == y->shape())
        << x->name() << " " << shapeToString(x)
        << " vs "
        << y->name() << " " << shapeToString(y);
}

template<typename T>
inline void ExpectEQ(const T& a, const T& b) {
    EXPECT_EQ(a, b);
}

template<>
inline void ExpectEQ<float>(const float& a, const float& b) {
    EXPECT_FLOAT_EQ(a, b);
}

template<>
inline void ExpectEQ<double>(const double& a, const double& b) {
    EXPECT_DOUBLE_EQ(a, b);
}

template<typename X_TYPE, typename Y_TYPE>
inline void ExpectTensorEQ(const Tensor* x, const Tensor* y) {
    ASSERT_TRUE(x->dataType() == y->dataType());
    ExpectDimsEQ(x, y);
    auto a = x->data<X_TYPE>();
    auto b = y->data<Y_TYPE>();
    for (int i = 0; i < x->elementSize(); ++i) {
        ExpectEQ(a[i], b[i]);
    }
}

} // namespace Test
} // namespace MAI
