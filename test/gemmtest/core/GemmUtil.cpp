#include <arm_neon.h>
#include "GemmUtil.h"

namespace MAI {
namespace Test {
void gemm_random(const float* A, const float* B, float* C, int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                C[m * N + n] += A[m * K + k] * B[k * N + n];
            }
        }
    }
}

void gemm_op2(const float* A, const float* B, float* C, int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; n += 4) {
            for (int k = 0; k < K; ++k) {
                C[m * N + n] += A[m * K + k] * B[k * N + n];
                C[m * N + n + 1] += A[m * K + k] * B[k * N + n + 1];
                C[m * N + n + 2] += A[m * K + k] * B[k * N + n + 2];
                C[m * N + n + 3] += A[m * K + k] * B[k * N + n + 3];
            }
        }
    }
}

void gemm_op3(const float* A, const float* B, float* C, int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    for (int m = 0; m < M; m += 4) {
        for (int n = 0; n < N; n += 4) {
            for (int k = 0; k < K; ++k) {
                C[m * N + n] += A[m * K + k] * B[k * N + n];
                C[m * N + n + 1] += A[m * K + k] * B[k * N + n + 1];
                C[m * N + n + 2] += A[m * K + k] * B[k * N + n + 2];
                C[m * N + n + 3] += A[m * K + k] * B[k * N + n + 3];
                C[(m + 1) * N + n]     += A[(m + 1) * K + k] * B[k * N + n];
                C[(m + 1) * N + n + 1] += A[(m + 1) * K + k] * B[k * N + n + 1];
                C[(m + 1) * N + n + 2] += A[(m + 1) * K + k] * B[k * N + n + 2];
                C[(m + 1) * N + n + 3] += A[(m + 1) * K + k] * B[k * N + n + 3];
                C[(m + 2) * N + n]     += A[(m + 2) * K + k] * B[k * N + n];
                C[(m + 2) * N + n + 1] += A[(m + 2) * K + k] * B[k * N + n + 1];
                C[(m + 2) * N + n + 2] += A[(m + 2) * K + k] * B[k * N + n + 2];
                C[(m + 2) * N + n + 3] += A[(m + 2) * K + k] * B[k * N + n + 3];
                C[(m + 3) * N + n]     += A[(m + 3) * K + k] * B[k * N + n];
                C[(m + 3) * N + n + 1] += A[(m + 3) * K + k] * B[k * N + n + 1];
                C[(m + 3) * N + n + 2] += A[(m + 3) * K + k] * B[k * N + n + 2];
                C[(m + 3) * N + n + 3] += A[(m + 3) * K + k] * B[k * N + n + 3];
            }
        }
    }
}

void gemm_op4(const float* A, const float* B, float* C, int M, int N, int K) {
#define GEMM_OP4_KERNEL(k) \
    C[m * N + n] += A[m * K + (k)] * B[(k) * N + n];                       \
    C[m * N + n + 1] += A[m * K + (k)] * B[(k) * N + n + 1];               \
    C[m * N + n + 2] += A[m * K + (k)] * B[(k) * N + n + 2];               \
    C[m * N + n + 3] += A[m * K + (k)] * B[(k) * N + n + 3];               \
    C[(m + 1) * N + n]     += A[(m + 1) * K + (k)] * B[(k) * N + n];       \
    C[(m + 1) * N + n + 1] += A[(m + 1) * K + (k)] * B[(k) * N + n + 1];   \
    C[(m + 1) * N + n + 2] += A[(m + 1) * K + (k)] * B[(k) * N + n + 2];   \
    C[(m + 1) * N + n + 3] += A[(m + 1) * K + (k)] * B[(k) * N + n + 3];   \
    C[(m + 2) * N + n]     += A[(m + 2) * K + (k)] * B[(k) * N + n];       \
    C[(m + 2) * N + n + 1] += A[(m + 2) * K + (k)] * B[(k) * N + n + 1];   \
    C[(m + 2) * N + n + 2] += A[(m + 2) * K + (k)] * B[(k) * N + n + 2];   \
    C[(m + 2) * N + n + 3] += A[(m + 2) * K + (k)] * B[(k) * N + n + 3];   \
    C[(m + 3) * N + n]     += A[(m + 3) * K + (k)] * B[(k) * N + n];       \
    C[(m + 3) * N + n + 1] += A[(m + 3) * K + (k)] * B[(k) * N + n + 1];   \
    C[(m + 3) * N + n + 2] += A[(m + 3) * K + (k)] * B[(k) * N + n + 2];   \
    C[(m + 3) * N + n + 3] += A[(m + 3) * K + (k)] * B[(k) * N + n + 3];   \

    memset(C, 0, M * N * sizeof(float));
    for (int m = 0; m < M; m += 4) {
        for (int n = 0; n < N; n += 4) {
            for (int k = 0; k < K; k += 4) {
                GEMM_OP4_KERNEL(k);
                GEMM_OP4_KERNEL(k + 1);
                GEMM_OP4_KERNEL(k + 2);
                GEMM_OP4_KERNEL(k + 3);
            }
        }
    }
}

void gemm_tile_884(const float* A, const float* B, float* C, int M, int N, int K) {
#define GEMM_TILE_884_ROW(CROW, AROW, AINDEX, BROW) \
    C[CROW * N + 0] += A[AROW * K + AINDEX] * B[BROW * N + 0]; \
    C[CROW * N + 1] += A[AROW * K + AINDEX] * B[BROW * N + 1]; \
    C[CROW * N + 2] += A[AROW * K + AINDEX] * B[BROW * N + 2]; \
    C[CROW * N + 3] += A[AROW * K + AINDEX] * B[BROW * N + 3]; \

#define GEMM_TILE_884_LOOP(AROW) \
    GEMM_TILE_884_ROW(AROW, AROW, 0, 0); \
    GEMM_TILE_884_ROW(AROW, AROW, 1, 1); \
    GEMM_TILE_884_ROW(AROW, AROW, 2, 2); \
    GEMM_TILE_884_ROW(AROW, AROW, 3, 3); \
    GEMM_TILE_884_ROW(AROW, AROW, 4, 4); \
    GEMM_TILE_884_ROW(AROW, AROW, 5, 5); \
    GEMM_TILE_884_ROW(AROW, AROW, 6, 6); \
    GEMM_TILE_884_ROW(AROW, AROW, 7, 7); \

#define GEMM_TILE_884_OUTER \
    GEMM_TILE_884_LOOP(0); \
    GEMM_TILE_884_LOOP(1); \
    GEMM_TILE_884_LOOP(2); \
    GEMM_TILE_884_LOOP(3); \
    GEMM_TILE_884_LOOP(4); \
    GEMM_TILE_884_LOOP(5); \
    GEMM_TILE_884_LOOP(6); \
    GEMM_TILE_884_LOOP(7); \

    GEMM_TILE_884_OUTER

}

void gemm_tile_884_neon_b_morton(const float* A, const float* B, float* C, int M, int N, int K) {
    float32x4_t c0 = vld1q_f32(C + 0 * N);
    float32x4_t c1 = vld1q_f32(C + 1 * N);
    float32x4_t c2 = vld1q_f32(C + 2 * N);
    float32x4_t c3 = vld1q_f32(C + 3 * N);
    float32x4_t c4 = vld1q_f32(C + 4 * N);
    float32x4_t c5 = vld1q_f32(C + 5 * N);
    float32x4_t c6 = vld1q_f32(C + 6 * N);
    float32x4_t c7 = vld1q_f32(C + 7 * N);

    float32x4_t a0 = vld1q_f32(A + 0 * K + 0);
    float32x4_t a1 = vld1q_f32(A + 0 * K + 4);
    float32x4_t a2 = vld1q_f32(A + 1 * K + 0);
    float32x4_t a3 = vld1q_f32(A + 1 * K + 4);
    float32x4_t a4 = vld1q_f32(A + 2 * K + 0);
    float32x4_t a5 = vld1q_f32(A + 2 * K + 4);
    float32x4_t a6 = vld1q_f32(A + 3 * K + 0);
    float32x4_t a7 = vld1q_f32(A + 3 * K + 4);
    float32x4_t a8 = vld1q_f32(A + 4 * K + 0);
    float32x4_t a9 = vld1q_f32(A + 4 * K + 4);
    float32x4_t a10 = vld1q_f32(A + 5 * K + 0);
    float32x4_t a11 = vld1q_f32(A + 5 * K + 4);
    float32x4_t a12 = vld1q_f32(A + 6 * K + 0);
    float32x4_t a13 = vld1q_f32(A + 6 * K + 4);
    float32x4_t a14 = vld1q_f32(A + 7 * K + 0);
    float32x4_t a15 = vld1q_f32(A + 7 * K + 4);

    float32x4_t b0 = vld1q_f32(B + 0 * 4);
    float32x4_t b1 = vld1q_f32(B + 1 * 4);
    float32x4_t b2 = vld1q_f32(B + 2 * 4);
    float32x4_t b3 = vld1q_f32(B + 3 * 4);
    float32x4_t b4 = vld1q_f32(B + 4 * 4);
    float32x4_t b5 = vld1q_f32(B + 5 * 4);
    float32x4_t b6 = vld1q_f32(B + 6 * 4);
    float32x4_t b7 = vld1q_f32(B + 7 * 4);

#define GEMM_TILE_884_FMAQ(cV, a0V, a1V) \
    cV = vfmaq_laneq_f32(cV, b0, a0V, 0); \
    cV = vfmaq_laneq_f32(cV, b1, a0V, 1); \
    cV = vfmaq_laneq_f32(cV, b2, a0V, 2); \
    cV = vfmaq_laneq_f32(cV, b3, a0V, 3); \
    cV = vfmaq_laneq_f32(cV, b4, a1V, 0); \
    cV = vfmaq_laneq_f32(cV, b5, a1V, 1); \
    cV = vfmaq_laneq_f32(cV, b6, a1V, 2); \
    cV = vfmaq_laneq_f32(cV, b7, a1V, 3); \

    GEMM_TILE_884_FMAQ(c0, a0, a1);
    GEMM_TILE_884_FMAQ(c1, a2, a3);
    GEMM_TILE_884_FMAQ(c2, a4, a5);
    GEMM_TILE_884_FMAQ(c3, a6, a7);
    GEMM_TILE_884_FMAQ(c4, a8, a9);
    GEMM_TILE_884_FMAQ(c5, a10, a11);
    GEMM_TILE_884_FMAQ(c6, a12, a13);
    GEMM_TILE_884_FMAQ(c7, a14, a15);

    vst1q_f32(C + 0 * N, c0);
    vst1q_f32(C + 1 * N, c1);
    vst1q_f32(C + 2 * N, c2);
    vst1q_f32(C + 3 * N, c3);
    vst1q_f32(C + 4 * N, c4);
    vst1q_f32(C + 5 * N, c5);
    vst1q_f32(C + 6 * N, c6);
    vst1q_f32(C + 7 * N, c7);
}

void gemm_tile_884_neon(const float* A, const float* B, float* C, int M, int N, int K) {
    float32x4_t c0 = vld1q_f32(C + 0 * N);
    float32x4_t c1 = vld1q_f32(C + 1 * N);
    float32x4_t c2 = vld1q_f32(C + 2 * N);
    float32x4_t c3 = vld1q_f32(C + 3 * N);
    float32x4_t c4 = vld1q_f32(C + 4 * N);
    float32x4_t c5 = vld1q_f32(C + 5 * N);
    float32x4_t c6 = vld1q_f32(C + 6 * N);
    float32x4_t c7 = vld1q_f32(C + 7 * N);

    float32x4_t a0 = vld1q_f32(A + 0 * K + 0);
    float32x4_t a1 = vld1q_f32(A + 0 * K + 4);
    float32x4_t a2 = vld1q_f32(A + 1 * K + 0);
    float32x4_t a3 = vld1q_f32(A + 1 * K + 4);
    float32x4_t a4 = vld1q_f32(A + 2 * K + 0);
    float32x4_t a5 = vld1q_f32(A + 2 * K + 4);
    float32x4_t a6 = vld1q_f32(A + 3 * K + 0);
    float32x4_t a7 = vld1q_f32(A + 3 * K + 4);
    float32x4_t a8 = vld1q_f32(A + 4 * K + 0);
    float32x4_t a9 = vld1q_f32(A + 4 * K + 4);
    float32x4_t a10 = vld1q_f32(A + 5 * K + 0);
    float32x4_t a11 = vld1q_f32(A + 5 * K + 4);
    float32x4_t a12 = vld1q_f32(A + 6 * K + 0);
    float32x4_t a13 = vld1q_f32(A + 6 * K + 4);
    float32x4_t a14 = vld1q_f32(A + 7 * K + 0);
    float32x4_t a15 = vld1q_f32(A + 7 * K + 4);

    float32x4_t b0 = vld1q_f32(B + 0 * N);
    float32x4_t b1 = vld1q_f32(B + 1 * N);
    float32x4_t b2 = vld1q_f32(B + 2 * N);
    float32x4_t b3 = vld1q_f32(B + 3 * N);
    float32x4_t b4 = vld1q_f32(B + 4 * N);
    float32x4_t b5 = vld1q_f32(B + 5 * N);
    float32x4_t b6 = vld1q_f32(B + 6 * N);
    float32x4_t b7 = vld1q_f32(B + 7 * N);

    GEMM_TILE_884_FMAQ(c0, a0, a1);
    GEMM_TILE_884_FMAQ(c1, a2, a3);
    GEMM_TILE_884_FMAQ(c2, a4, a5);
    GEMM_TILE_884_FMAQ(c3, a6, a7);
    GEMM_TILE_884_FMAQ(c4, a8, a9);
    GEMM_TILE_884_FMAQ(c5, a10, a11);
    GEMM_TILE_884_FMAQ(c6, a12, a13);
    GEMM_TILE_884_FMAQ(c7, a14, a15);

    vst1q_f32(C + 0 * N, c0);
    vst1q_f32(C + 1 * N, c1);
    vst1q_f32(C + 2 * N, c2);
    vst1q_f32(C + 3 * N, c3);
    vst1q_f32(C + 4 * N, c4);
    vst1q_f32(C + 5 * N, c5);
    vst1q_f32(C + 6 * N, c6);
    vst1q_f32(C + 7 * N, c7);
}

void gemm_block(const float* A, const float* B, float* C, int MB, int NB, int KB,
        int M, int N, int K) {
    constexpr int MT = 8;
    constexpr int NT = 4;
    constexpr int KT = 8;
    for (int m = 0; m < MB; m += MT) {
        for (int k = 0; k < KB; k += KT) {
            for (int n = 0; n < NB; n += NT) {
                gemm_tile_884(A + (m * K + k),
                        B + (k * N + n),
                        C + (m * N + n),
                        M , N, K);
            }
        }
    }
}

void gemm_op5(const float* A, const float* B, float* C, int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    constexpr int MB = 8;
    constexpr int NB = 4;
    constexpr int KB = 8;
    for (int m = 0; m < M; m += MB) {
        for (int k = 0; k < K; k += KB) {
            for (int n = 0; n < N; n += NB) {
                gemm_block(A + (m * K + k),
                        B + (k * N + n),
                        C + (m * N + n),
                        MB, NB, KB, M , N, K);
            }
        }
    }
}
void gemm_block_neon(const float* A, const float* B, float* C, int MB, int NB, int KB,
        int M, int N, int K) {
    constexpr int MT = 8;
    constexpr int NT = 4;
    constexpr int KT = 8;
    for (int m = 0; m < MB; m += MT) {
        for (int k = 0; k < KB; k += KT) {
            for (int n = 0; n < NB; n += NT) {
                gemm_tile_884_neon(A + (m * K + k),
                        B + (k * N + n),
                        C + (m * N + n),
                        M , N, K);
            }
        }
    }
}

void gemm_op6(const float* A, const float* B, float* C, int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    constexpr int MB = 64;
    constexpr int NB = 64;
    constexpr int KB = 64;
    for (int m = 0; m < M; m += MB) {
        for (int k = 0; k < K; k += KB) {
            for (int n = 0; n < N; n += NB) {
                gemm_block_neon(A + (m * K + k),
                        B + (k * N + n),
                        C + (m * N + n),
                        MB, NB, KB, M , N, K);
            }
        }
    }
}

void pack(float* packedData, const float* A, int MT, int KT, int K) {
    for (int mt = 0; mt < MT; ++mt) {
        memcpy(packedData + mt * KT, A + mt * K, KT * sizeof(float));
    }
}

void packB(float* packedData, const float* B, int KT, int NT, int K) {
    for (int kt = 0; kt < KT; ++kt) {
        for (int nt = 0; nt < NT; nt += 4) {
            memcpy(packedData + kt * 4 + nt * 8, B + kt * K + nt, 4 * sizeof(float));
        }
    }
}

void gemm_block_packed_A(const float* A, const float* B, float* C, int MB, int NB, int KB,
        int M, int N, int K) {
    constexpr int MT = 8;
    constexpr int NT = 4;
    constexpr int KT = 8;
    float packedA[8 * 8];
    for (int m = 0; m < MB; m += MT) {
        for (int k = 0; k < KB; k += KT) {
            pack(packedA, A + m * K + k, MT, KT, K);
            for (int n = 0; n < NB; n += NT) {
                gemm_tile_884_neon(packedA,
                        B + (k * N + n),
                        C + (m * N + n),
                        M , N, 8);
            }
        }
    }
}

void gemm_block_packed_A_packed_B(const float* A, const float* B, float* C, int MB, int NB, int KB,
        int M, int N, int K) {
    constexpr int MT = 8;
    constexpr int NT = 4;
    constexpr int KT = 8;
    float packedA[8 * 8];
    float packedB[64 * 64];
    for (int m = 0; m < MB; m += MT) {
        for (int k = 0; k < KB; k += KT) {
            pack(packedA, A + m * K + k, MT, KT, K);
            if (m == 0) {
                packB(packedB + k * 64, B + k * N, KT, 64, K);
            }
            for (int n = 0; n < NB; n += NT) {
                gemm_tile_884_neon_b_morton(packedA,
                        packedB + (k * 64 + n * 8),
                        C + (m * N + n),
                        M , N, 8);
            }
        }
    }
}

void gemm_op7(const float* A, const float* B, float* C, int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    constexpr int MB = 64;
    constexpr int NB = 64;
    constexpr int KB = 64;
    for (int m = 0; m < M; m += MB) {
        for (int k = 0; k < K; k += KB) {
            for (int n = 0; n < N; n += NB) {
                gemm_block_packed_A(A + (m * K + k),
                        B + (k * N + n),
                        C + (m * N + n),
                        MB, NB, KB, M , N, K);
            }
        }
    }
}
void gemm_op8(const float* A, const float* B, float* C, int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    constexpr int MB = 64;
    constexpr int NB = 64;
    constexpr int KB = 64;
#pragma omp parallel for schedule(dynamic) collapse(2)
    for (int m = 0; m < M; m += MB) {
        for (int n = 0; n < N; n += NB) {
            for (int k = 0; k < K; k += KB) {
                gemm_block_packed_A_packed_B(A + (m * K + k),
                        B + (k * N + n),
                        C + (m * N + n),
                        MB, NB, KB, M , N, K);
            }
        }
    }
}
} // namespace Test
} // namespace MAI
