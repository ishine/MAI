#pragma once
#include <arm_neon.h>

namespace MAI {
namespace Test {

template<bool rowMajor, bool transA, bool transB>
void sgemm(int M, int N, int K,
        float alpha,
        const float* A, int lda,
        const float* B, int ldb,
        float beta,
        float* C, int ldc);

template<bool rowMajor, bool transA, bool transB>
void sgemm_op(int M, int N, int K,
        float alpha,
        const float* A, int lda,
        const float* B, int ldb,
        float beta,
        float* C, int ldc);

template<>
void sgemm<true, true, true>(int M, int N, int K,
        float alpha,
        const float* A, int lda,
        const float* B, int ldb,
        float beta,
        float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            C[m * ldc + n] *= beta;
            for (int k = 0; k < K; ++k) {
                C[m * ldc + n] += alpha * A[k * lda + m] * B[n * ldb + k];

            }
        }
    }
}

} // namespace Test
} // namespace MAI
