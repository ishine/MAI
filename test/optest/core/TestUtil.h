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
    auto b = x->data<Y_TYPE>();
    for (int i = 0; i < x->elementSize(); ++i) {
        ExpectEQ(a[i], b[i]);
    }
}

} // namespace Test
} // namespace MAI
