#pragma once
#include <random>
#include <sys/time.h>
#include "GemmUtil.h"
namespace MAI {
namespace Test {

void sgemm(bool rowMajor, bool transA, bool transB,
        int M, int N, int K,
        float alpha,
        const float* A, int lda,
        const float* B, int ldb,
        float beta,
        float* C, int ldc);

void sgemm_op(bool rowMajor, bool transA, bool transB,
        int M, int N, int K,
        float alpha,
        const float* A, int lda,
        const float* B, int ldb,
        float beta,
        float* C, int ldc);

} // namespace Test
} // namespace MAI
