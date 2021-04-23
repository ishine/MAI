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
void sgemm<true, false, true>(int M, int N, int K,
        float alpha,
        const float* A, int lda,
        const float* B, int ldb,
        float beta,
        float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            C[m * ldc + n] *= beta;
            for (int k = 0; k < K; ++k) {
                C[m * ldc + n] += alpha * A[m * lda + k] * B[n * ldb + k];

            }
        }
    }
}

void sgemm_rnt_tile_448(
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
            for (int k = 0; k < 8; ++k) {
                C[m * ldc + n] += alpha * A[m * lda + k] * B[n * ldb + k];

            }
        }
    }
}

#define SGEMM_RNT_448_MUL(c0, a0, a1) \
    r0 = vfmaq_f32(vmulq_f32(a0, b0), a1, b1); \
    r1 = vfmaq_f32(vmulq_f32(a0, b2), a1, b3); \
    r2 = vpaddq_f32(r0, r1);                   \
    r0 = vfmaq_f32(vmulq_f32(a0, b4), a1, b5); \
    r1 = vfmaq_f32(vmulq_f32(a0, b6), a1, b7); \
    r3 = vpaddq_f32(r0, r1);                   \
    r2 = vpaddq_f32(r2, r3);                   \
    c0 = vaddq_f32(vmulq_f32(r2, va), vmulq_f32(c0, vb));

void sgemm_rnt_tile_448_neon(
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
    float32x4_t a0 = vld1q_f32(A + lda * 0 + 0);
    float32x4_t a1 = vld1q_f32(A + lda * 0 + 4);
    float32x4_t a2 = vld1q_f32(A + lda * 1 + 0);
    float32x4_t a3 = vld1q_f32(A + lda * 1 + 4);
    float32x4_t a4 = vld1q_f32(A + lda * 2 + 0);
    float32x4_t a5 = vld1q_f32(A + lda * 2 + 4);
    float32x4_t a6 = vld1q_f32(A + lda * 3 + 0);
    float32x4_t a7 = vld1q_f32(A + lda * 3 + 4);

    float32x4_t b0 = vld1q_f32(B + ldb * 0 + 0);
    float32x4_t b1 = vld1q_f32(B + ldb * 0 + 4);
    float32x4_t b2 = vld1q_f32(B + ldb * 1 + 0);
    float32x4_t b3 = vld1q_f32(B + ldb * 1 + 4);
    float32x4_t b4 = vld1q_f32(B + ldb * 2 + 0);
    float32x4_t b5 = vld1q_f32(B + ldb * 2 + 4);
    float32x4_t b6 = vld1q_f32(B + ldb * 3 + 0);
    float32x4_t b7 = vld1q_f32(B + ldb * 3 + 4);

    float32x4_t c0 = vld1q_f32(C + ldc * 0);
    float32x4_t c1 = vld1q_f32(C + ldc * 1);
    float32x4_t c2 = vld1q_f32(C + ldc * 2);
    float32x4_t c3 = vld1q_f32(C + ldc * 3);

    float32x4_t r0, r1, r2, r3;
    SGEMM_RNT_448_MUL(c0, a0, a1);
    SGEMM_RNT_448_MUL(c1, a2, a3);
    SGEMM_RNT_448_MUL(c2, a4, a5);
    SGEMM_RNT_448_MUL(c3, a6, a7);

    vst1q_f32(C + ldc * 0, c0);
    vst1q_f32(C + ldc * 1, c1);
    vst1q_f32(C + ldc * 2, c2);
    vst1q_f32(C + ldc * 3, c3);
}

void sgemm_rnt_tile_848_neon(
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
    float32x4_t a0 = vld1q_f32(A + lda * 0 + 0);
    float32x4_t a1 = vld1q_f32(A + lda * 0 + 4);
    float32x4_t a2 = vld1q_f32(A + lda * 1 + 0);
    float32x4_t a3 = vld1q_f32(A + lda * 1 + 4);
    float32x4_t a4 = vld1q_f32(A + lda * 2 + 0);
    float32x4_t a5 = vld1q_f32(A + lda * 2 + 4);
    float32x4_t a6 = vld1q_f32(A + lda * 3 + 0);
    float32x4_t a7 = vld1q_f32(A + lda * 3 + 4);
    float32x4_t a8 = vld1q_f32(A + lda * 4 + 0);
    float32x4_t a9 = vld1q_f32(A + lda * 4 + 4);
    float32x4_t a10 = vld1q_f32(A + lda * 5 + 0);
    float32x4_t a11 = vld1q_f32(A + lda * 5 + 4);
    float32x4_t a12 = vld1q_f32(A + lda * 6 + 0);
    float32x4_t a13 = vld1q_f32(A + lda * 6 + 4);
    float32x4_t a14 = vld1q_f32(A + lda * 7 + 0);
    float32x4_t a15 = vld1q_f32(A + lda * 7 + 4);

    float32x4_t b0 = vld1q_f32(B + ldb * 0 + 0);
    float32x4_t b1 = vld1q_f32(B + ldb * 0 + 4);
    float32x4_t b2 = vld1q_f32(B + ldb * 1 + 0);
    float32x4_t b3 = vld1q_f32(B + ldb * 1 + 4);
    float32x4_t b4 = vld1q_f32(B + ldb * 2 + 0);
    float32x4_t b5 = vld1q_f32(B + ldb * 2 + 4);
    float32x4_t b6 = vld1q_f32(B + ldb * 3 + 0);
    float32x4_t b7 = vld1q_f32(B + ldb * 3 + 4);

    float32x4_t c0 = vld1q_f32(C + ldc * 0);
    float32x4_t c1 = vld1q_f32(C + ldc * 1);
    float32x4_t c2 = vld1q_f32(C + ldc * 2);
    float32x4_t c3 = vld1q_f32(C + ldc * 3);
    float32x4_t c4 = vld1q_f32(C + ldc * 4);
    float32x4_t c5 = vld1q_f32(C + ldc * 5);
    float32x4_t c6 = vld1q_f32(C + ldc * 6);
    float32x4_t c7 = vld1q_f32(C + ldc * 7);

    float32x4_t r0, r1, r2, r3;
    SGEMM_RNT_448_MUL(c0, a0, a1);
    SGEMM_RNT_448_MUL(c1, a2, a3);
    SGEMM_RNT_448_MUL(c2, a4, a5);
    SGEMM_RNT_448_MUL(c3, a6, a7);
    SGEMM_RNT_448_MUL(c4, a8, a9);
    SGEMM_RNT_448_MUL(c5, a10, a11);
    SGEMM_RNT_448_MUL(c6, a12, a13);
    SGEMM_RNT_448_MUL(c7, a14, a15);

    vst1q_f32(C + ldc * 0, c0);
    vst1q_f32(C + ldc * 1, c1);
    vst1q_f32(C + ldc * 2, c2);
    vst1q_f32(C + ldc * 3, c3);
    vst1q_f32(C + ldc * 4, c4);
    vst1q_f32(C + ldc * 5, c5);
    vst1q_f32(C + ldc * 6, c6);
    vst1q_f32(C + ldc * 7, c7);
}

void sgemm_rnt_block(
        int MB, int NB, int KB,
        float alpha,
        const float* A, int lda,
        const float* B, int ldb,
        float beta,
        float* C, int ldc) {
    constexpr int MT = 4;
    constexpr int NT = 4;
    constexpr int KT = 8;
    for (int m = 0; m < MB; m += MT) {
        for (int n = 0; n < NB; n += NT) {
            for (int k = 0; k < KB; k += KT) {
                sgemm_rnt_tile_448_neon(
                        alpha,
                        A + m * lda + k,
                        lda,
                        B + n * ldb + k,
                        ldb,
                        beta,
                        C + m * ldc + n,
                        ldc);
            }
        }
    }
}

static void pack(float* packData, const float* input,
        int lda, int tileR, int tileC, int row, int col) {
    int RB = row / tileR;
    int CB = col / tileC;
    for (int r = 0; r < RB; ++r) {
        for (int c = 0; c < CB; ++c) {
            int offsetPack = r * tileR * row + c * tileR * tileC;
            int offsetInput = r * tileR * lda + c * tileC;
            for (int t = 0; t < tileR; ++t) {
                memcpy(packData + offsetPack + t * tileC,
                        input + offsetInput + t * lda,
                        tileC * sizeof(float));
            }
        }
    }
}

void sgemm_rnt_block_packed_A_packed_B(
        int MB, int NB, int KB,
        float alpha,
        const float* A, int lda,
        const float* B, int ldb,
        float beta,
        float* C, int ldc) {
    constexpr int MT = 4;
    constexpr int NT = 4;
    constexpr int KT = 8;
    float packedA[4 * 64];
    float packedB[64 * 64];
    for (int m = 0; m < MB; m += MT) {
        pack(packedA, A + m * lda, lda, MT, KT, MT, 64);
        for (int n = 0; n < NB; n += NT) {
            if (m == 0) {
                pack(packedB + n * 64, B + n * ldb,
                        ldb, NT, KT, NT, 64);
            }
            for (int k = 0; k < KB; k += KT) {
               sgemm_rnt_tile_448_neon(
                       alpha,
                       // A + m * lda + k,
                       // lda,
                       packedA + k * MT,
                       KT,
                       // B + n * ldb + k,
                       // ldb,
                       packedB + n * 64 + k * NT,
                       KT,
                       beta,
                       C + m * ldc + n,
                       ldc);
            }
        }
    }
}

template<>
void sgemm_op<true, false, true>(
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
                sgemm_rnt_block_packed_A_packed_B(
                        MB, NB, KB,
                        alpha,
                        A + (m * MB * lda + k * KB),
                        lda,
                        B + (n * NB * ldb + k * KB),
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
