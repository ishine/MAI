#include "SGemmUtil.h"
#include "SGemmR_NN.h"
#include "SGemmR_NT.h"
#include "SGemmR_TN.h"
#include "SGemmR_TT.h"
#include <arm_neon.h>
namespace MAI {
namespace Test {

void sgemm(bool rowMajor, bool transA, bool transB,
        int M, int N, int K,
        float alpha,
        const float* A, int lda,
        const float* B, int ldb,
        float beta,
        float* C, int ldc) {
    if (rowMajor) {
        if (!transA && !transB) {
            sgemm<true, false, false>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } else if (!transA && transB) {
            sgemm<true, false, true>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } else if (transA && !transB) {
            sgemm<true, true, false>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } else if (transA && transB) {
            sgemm<true, true, true>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }
    }
}

void sgemm_op(bool rowMajor, bool transA, bool transB,
        int M, int N, int K,
        float alpha,
        const float* A, int lda,
        const float* B, int ldb,
        float beta,
        float* C, int ldc) {
    if (rowMajor) {
        if (!transA && !transB) {
            //sgemm<true, false, false>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } else if (!transA && transB) {
            sgemm_op<true, false, true>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } else if (transA && !transB) {
            sgemm_op<true, true, false>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } else if (transA && transB) {
            //sgemm<true, true, true>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }
    }
}

} // namespace Test
} // namespace MAI
