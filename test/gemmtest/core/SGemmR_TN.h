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
void sgemm<true, true, false>(int M, int N, int K,
        float alpha,
        const float* A, int lda,
        const float* B, int ldb,
        float beta,
        float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            C[m * ldc + n] *= beta;
            for (int k = 0; k < K; ++k) {
                C[m * ldc + n] += alpha * A[k * lda + m] * B[k * ldb + n];

            }
        }
    }
}

void sgemm_rtn_tile_444(
        float alpha,
        const float* A,
        int lda,
        const float* B,
        int ldb,
        float beta,
        float* C,
        int ldc) {
    for (int m = 0; m < 4; ++m) {
        for (int n = 0; n < 4; ++n) {
            C[m * ldc + n] *= beta;
            for (int k = 0; k < 4; ++k) {
                C[m * ldc + n] += alpha * A[k * lda + m] * B[k * ldb + n];

            }
        }
    }
}

void sgemm_rtn_tile_444_neon(
        float alpha,
        const float* A,
        int lda,
        const float* B,
        int ldb,
        float beta,
        float* C,
        int ldc) {
    float32x4_t va = vdupq_n_f32(alpha);
    float32x4_t vb = vdupq_n_f32(beta);
    float32x4_t a0 = vld1q_f32(A + lda * 0);
    float32x4_t a1 = vld1q_f32(A + lda * 1);
    float32x4_t a2 = vld1q_f32(A + lda * 2);
    float32x4_t a3 = vld1q_f32(A + lda * 3);

    float32x4_t b0 = vld1q_f32(B + ldb * 0);
    float32x4_t b1 = vld1q_f32(B + ldb * 1);
    float32x4_t b2 = vld1q_f32(B + ldb * 2);
    float32x4_t b3 = vld1q_f32(B + ldb * 3);

    float32x4_t c0 = vld1q_f32(C + ldc * 0);
    float32x4_t c1 = vld1q_f32(C + ldc * 1);
    float32x4_t c2 = vld1q_f32(C + ldc * 2);
    float32x4_t c3 = vld1q_f32(C + ldc * 3);

#define SGEMM_RTN_444_MUL(index) \
    c##index = vmulq_f32(c##index, vb); \
    c##index = vfmaq_laneq_f32(c##index, b0, a0, index); \
    c##index = vfmaq_laneq_f32(c##index, b1, a1, index); \
    c##index = vfmaq_laneq_f32(c##index, b2, a2, index); \
    c##index = vfmaq_laneq_f32(c##index, b3, a3, index); \

    SGEMM_RTN_444_MUL(0);
    SGEMM_RTN_444_MUL(1);
    SGEMM_RTN_444_MUL(2);
    SGEMM_RTN_444_MUL(3);

    vst1q_f32(C + 0 * ldc, c0);
    vst1q_f32(C + 1 * ldc, c1);
    vst1q_f32(C + 2 * ldc, c2);
    vst1q_f32(C + 3 * ldc, c3);
#undef SGEMM_RTN_444_MUL
}

void sgemm_rtn_tile_448_neon(
        float alpha,
        const float* A,
        int lda,
        const float* B,
        int ldb,
        float beta,
        float* C,
        int ldc) {
    float32x4_t va = vdupq_n_f32(alpha);
    float32x4_t vb = vdupq_n_f32(beta);
    float32x4_t a0 = vld1q_f32(A + lda * 0);
    float32x4_t a1 = vld1q_f32(A + lda * 1);
    float32x4_t a2 = vld1q_f32(A + lda * 2);
    float32x4_t a3 = vld1q_f32(A + lda * 3);
    float32x4_t a4 = vld1q_f32(A + lda * 4);
    float32x4_t a5 = vld1q_f32(A + lda * 5);
    float32x4_t a6 = vld1q_f32(A + lda * 6);
    float32x4_t a7 = vld1q_f32(A + lda * 7);

    float32x4_t b0 = vld1q_f32(B + ldb * 0);
    float32x4_t b1 = vld1q_f32(B + ldb * 1);
    float32x4_t b2 = vld1q_f32(B + ldb * 2);
    float32x4_t b3 = vld1q_f32(B + ldb * 3);
    float32x4_t b4 = vld1q_f32(B + ldb * 4);
    float32x4_t b5 = vld1q_f32(B + ldb * 5);
    float32x4_t b6 = vld1q_f32(B + ldb * 6);
    float32x4_t b7 = vld1q_f32(B + ldb * 7);

    float32x4_t c0 = vld1q_f32(C + ldc * 0);
    float32x4_t c1 = vld1q_f32(C + ldc * 1);
    float32x4_t c2 = vld1q_f32(C + ldc * 2);
    float32x4_t c3 = vld1q_f32(C + ldc * 3);

#define SGEMM_RTN_448_MUL(index) \
    c##index = vmulq_f32(c##index, vb); \
    c##index = vfmaq_laneq_f32(c##index, b0, a0, index); \
    c##index = vfmaq_laneq_f32(c##index, b1, a1, index); \
    c##index = vfmaq_laneq_f32(c##index, b2, a2, index); \
    c##index = vfmaq_laneq_f32(c##index, b3, a3, index); \
    c##index = vfmaq_laneq_f32(c##index, b4, a4, index); \
    c##index = vfmaq_laneq_f32(c##index, b5, a5, index); \
    c##index = vfmaq_laneq_f32(c##index, b6, a6, index); \
    c##index = vfmaq_laneq_f32(c##index, b7, a7, index); \

    SGEMM_RTN_448_MUL(0);
    SGEMM_RTN_448_MUL(1);
    SGEMM_RTN_448_MUL(2);
    SGEMM_RTN_448_MUL(3);

    vst1q_f32(C + 0 * ldc, c0);
    vst1q_f32(C + 1 * ldc, c1);
    vst1q_f32(C + 2 * ldc, c2);
    vst1q_f32(C + 3 * ldc, c3);
#undef SGEMM_RTN_448_MUL
}

void sgemm_rtn_block(
        int MB, int NB, int KB,
        float alpha,
        const float* A, int lda,
        const float* B, int ldb,
        float beta,
        float* C, int ldc) {
    constexpr int MT = 4;
    constexpr int NT = 4;
    constexpr int KT = 8;
    for (int k = 0; k < KB; k += KT) {
        for (int m = 0; m < MB; m += MT) {
            for (int n = 0; n < NB; n += NT) {
                sgemm_rtn_tile_448_neon(
                        alpha,
                        A + k * lda + m,
                        lda,
                        B + k * ldb + n,
                        ldb,
                        beta,
                        C + m * ldc + n,
                        ldc);
            }
        }
    }
}

void sgemm_rtn_block_packed_A_packed_B(
        int MB, int NB, int KB,
        float alpha,
        const float* A, int lda,
        const float* B, int ldb,
        float beta,
        float* C, int ldc) {
    constexpr int MT = 4;
    constexpr int NT = 4;
    constexpr int KT = 8;
    //float packedA[8 * 64];
    float packedB[8 * 64];
    for (int k = 0; k < KB; k += KT) {
        pack(packedB, B + k * ldb, ldb, KT, NT, KT, 64);
        for (int m = 0; m < MB; m += MT) {
            for (int n = 0; n < NB; n += NT) {
               sgemm_rtn_tile_448_neon(
                       alpha,
                       A + k * lda + m,
                       lda,
                       // B + k * ldb + n,
                       // ldb,
                       packedB + n * KT,
                       NT,
                       beta,
                       C + m * ldc + n,
                       ldc);
            }
        }
    }
}

template<>
void sgemm_op<true, true, false>(
        int M, int N, int K,
        float alpha,
        const float* A, int lda,
        const float* B, int ldb,
        float beta,
        float* C, int ldc) {
    constexpr int MB = 64;
    constexpr int NB = 64;
    constexpr int KB = 64;
    int MBCount = M / MB;
    int NBCount = N / NB;
    int KBCount = K / KB;
#pragma omp parallel for schedule(dynamic) collapse(2)
    for (int m = 0; m < MBCount; ++m) {
        for (int n = 0; n < NBCount; ++n) {
            for (int k = 0; k < KBCount; ++k) {
                sgemm_rtn_block_packed_A_packed_B(
                        MB, NB, KB,
                        alpha,
                        A + (k * KB * lda + m * MB),
                        lda,
                        B + (k * KB * ldb + n * NB),
                        ldb,
                        beta,
                        C + (m * MB * ldc + n * NB),
                        ldc);
            }
        }
    }
}

} // namespace Test
} // namespace MAI
