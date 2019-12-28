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

#include <arm_neon.h>

namespace MAI {
namespace Op {
namespace CPU {
namespace NEON {
#define MAI_VFMAQ_LANEQ_F32_4(OValue, AValue)             \
    OValue = vfmaq_laneq_f32(OValue, b0Value, AValue, 0); \
    OValue = vfmaq_laneq_f32(OValue, b1Value, AValue, 1); \
    OValue = vfmaq_laneq_f32(OValue, b2Value, AValue, 2); \
    OValue = vfmaq_laneq_f32(OValue, b3Value, AValue, 3); \

#define MAI_VFMAQ_LANEQ_F32_8(OValue, A0Value, A1Value)             \
    OValue = vfmaq_laneq_f32(OValue, b0Value, A0Value, 0); \
    OValue = vfmaq_laneq_f32(OValue, b1Value, A0Value, 1); \
    OValue = vfmaq_laneq_f32(OValue, b2Value, A0Value, 2); \
    OValue = vfmaq_laneq_f32(OValue, b3Value, A0Value, 3); \
    OValue = vfmaq_laneq_f32(OValue, b4Value, A1Value, 0); \
    OValue = vfmaq_laneq_f32(OValue, b5Value, A1Value, 1); \
    OValue = vfmaq_laneq_f32(OValue, b6Value, A1Value, 2); \
    OValue = vfmaq_laneq_f32(OValue, b7Value, A1Value, 3); \

template<typename T, bool transpose_a, bool transpose_b>
struct Gemm;

template<>
struct Gemm<float, false, false> {

    static void gemm_tile_444(const float* aPtr, const float* bPtr, float* oPtr, int M, int N, int K) {
        float32x4_t o0Value = vld1q_f32(oPtr);
        float32x4_t o1Value = vld1q_f32(oPtr + N);
        float32x4_t o2Value = vld1q_f32(oPtr + N * 2);
        float32x4_t o3Value = vld1q_f32(oPtr + N * 3);

        float32x4_t a0Value = vld1q_f32(aPtr);
        float32x4_t a1Value = vld1q_f32(aPtr + K);
        float32x4_t a2Value = vld1q_f32(aPtr + K * 2);
        float32x4_t a3Value = vld1q_f32(aPtr + K * 3);

        float32x4_t b0Value = vld1q_f32(bPtr);
        float32x4_t b1Value = vld1q_f32(bPtr + N);
        float32x4_t b2Value = vld1q_f32(bPtr + N * 2);
        float32x4_t b3Value = vld1q_f32(bPtr + N * 3);

        o0Value = vfmaq_laneq_f32(o0Value, b0Value, a0Value, 0);
        o0Value = vfmaq_laneq_f32(o0Value, b1Value, a0Value, 1);
        o0Value = vfmaq_laneq_f32(o0Value, b2Value, a0Value, 2);
        o0Value = vfmaq_laneq_f32(o0Value, b3Value, a0Value, 3);

        o1Value = vfmaq_laneq_f32(o1Value, b0Value, a1Value, 0);
        o1Value = vfmaq_laneq_f32(o1Value, b1Value, a1Value, 1);
        o1Value = vfmaq_laneq_f32(o1Value, b2Value, a1Value, 2);
        o1Value = vfmaq_laneq_f32(o1Value, b3Value, a1Value, 3);

        o2Value = vfmaq_laneq_f32(o2Value, b0Value, a2Value, 0);
        o2Value = vfmaq_laneq_f32(o2Value, b1Value, a2Value, 1);
        o2Value = vfmaq_laneq_f32(o2Value, b2Value, a2Value, 2);
        o2Value = vfmaq_laneq_f32(o2Value, b3Value, a2Value, 3);

        o3Value = vfmaq_laneq_f32(o3Value, b0Value, a3Value, 0);
        o3Value = vfmaq_laneq_f32(o3Value, b1Value, a3Value, 1);
        o3Value = vfmaq_laneq_f32(o3Value, b2Value, a3Value, 2);
        o3Value = vfmaq_laneq_f32(o3Value, b3Value, a3Value, 3);

        vst1q_f32(oPtr, o0Value);
        vst1q_f32(oPtr + N, o1Value);
        vst1q_f32(oPtr + N * 2, o2Value);
        vst1q_f32(oPtr + N * 3, o3Value);
    }

    // (M, K, N) ==> (8, 8, 4)
    static void gemm_tile_884(const float* aPtr, const float* bPtr, float* oPtr, int M, int N, int K) {
        float32x4_t o0Value = vld1q_f32(oPtr);
        float32x4_t o1Value = vld1q_f32(oPtr + N);
        float32x4_t o2Value = vld1q_f32(oPtr + N * 2);
        float32x4_t o3Value = vld1q_f32(oPtr + N * 3);
        float32x4_t o4Value = vld1q_f32(oPtr + N * 4);
        float32x4_t o5Value = vld1q_f32(oPtr + N * 5);
        float32x4_t o6Value = vld1q_f32(oPtr + N * 6);
        float32x4_t o7Value = vld1q_f32(oPtr + N * 7);

        float32x4_t a0Value = vld1q_f32(aPtr);
        float32x4_t a1Value = vld1q_f32(aPtr + 4);
        float32x4_t a2Value = vld1q_f32(aPtr + K);
        float32x4_t a3Value = vld1q_f32(aPtr + K + 4);
        float32x4_t a4Value = vld1q_f32(aPtr + K * 2);
        float32x4_t a5Value = vld1q_f32(aPtr + K * 2 + 4);
        float32x4_t a6Value = vld1q_f32(aPtr + K * 3);
        float32x4_t a7Value = vld1q_f32(aPtr + K * 3 + 4);
        float32x4_t a8Value = vld1q_f32(aPtr + K * 4);
        float32x4_t a9Value = vld1q_f32(aPtr + K * 4 + 4);
        float32x4_t a10Value = vld1q_f32(aPtr + K * 5);
        float32x4_t a11Value = vld1q_f32(aPtr + K * 5 + 4);
        float32x4_t a12Value = vld1q_f32(aPtr + K * 6);
        float32x4_t a13Value = vld1q_f32(aPtr + K * 6 + 4);
        float32x4_t a14Value = vld1q_f32(aPtr + K * 7);
        float32x4_t a15Value = vld1q_f32(aPtr + K * 7 + 4);

        float32x4_t b0Value = vld1q_f32(bPtr);
        float32x4_t b1Value = vld1q_f32(bPtr + N);
        float32x4_t b2Value = vld1q_f32(bPtr + N * 2);
        float32x4_t b3Value = vld1q_f32(bPtr + N * 3);
        float32x4_t b4Value = vld1q_f32(bPtr + N * 4);
        float32x4_t b5Value = vld1q_f32(bPtr + N * 5);
        float32x4_t b6Value = vld1q_f32(bPtr + N * 6);
        float32x4_t b7Value = vld1q_f32(bPtr + N * 7);

        MAI_VFMAQ_LANEQ_F32_8(o0Value, a0Value, a1Value);
        MAI_VFMAQ_LANEQ_F32_8(o1Value, a2Value, a3Value);
        MAI_VFMAQ_LANEQ_F32_8(o2Value, a4Value, a5Value);
        MAI_VFMAQ_LANEQ_F32_8(o3Value, a6Value, a7Value);
        MAI_VFMAQ_LANEQ_F32_8(o4Value, a8Value, a9Value);
        MAI_VFMAQ_LANEQ_F32_8(o5Value, a10Value, a11Value);
        MAI_VFMAQ_LANEQ_F32_8(o6Value, a12Value, a13Value);
        MAI_VFMAQ_LANEQ_F32_8(o7Value, a14Value, a15Value);

        vst1q_f32(oPtr, o0Value);
        vst1q_f32(oPtr + N, o1Value);
        vst1q_f32(oPtr + N * 2, o2Value);
        vst1q_f32(oPtr + N * 3, o3Value);
        vst1q_f32(oPtr + N * 4, o4Value);
        vst1q_f32(oPtr + N * 5, o5Value);
        vst1q_f32(oPtr + N * 6, o6Value);
        vst1q_f32(oPtr + N * 7, o7Value);
    }

    static void gemm_random_block(const float* aPtr, const float* bPtr, float* oPtr,
            int blockM, int blockN, int blockK, int M, int N, int K) {
        for (int m = 0; m < blockM; ++m) {
            for (int k = 0; k < blockK; ++k) {
                for (int n = 0; n < blockN; ++n) {
                    oPtr[m * N + n] += aPtr[m * K + k] * bPtr[k * N + n];
                }
            }
        }
    }

    static void pack(const float* bPtr, float* packedBPtr, int blockM, int blockN, int stride, int tile) {
        for(int i = 0; i < blockN; i += tile) {
            for (int j = 0; j < blockM; ++j) {
                memcpy(packedBPtr + i * blockM + j * tile, bPtr + j * stride + i, sizeof(float) * tile);
            }
        }
    }

    template<int MAX_BLOCKM, int MAX_BLOCKK, int MAX_BLOCKN>
    static void gemm_tile(const float* aPtr, const float* bPtr, float* oPtr,
            const int blockM, const int blockN, const int blockK, int M, int N, int K) {
        const int tileMSize = 8;
        const int tileKSize = 8;
        const int tileNSize = 4;
#if !defined(MAI_GEMM_NO_CACHE)
        float packedB[MAX_BLOCKK * MAX_BLOCKN];
#endif

#if !defined(MAI_GEMM_NO_CACHE) && !defined(MAI_GEMM_CACHE_B)
        float packedA[8 * 8];
#endif

#ifdef MAI_GEMM_CACHE_ABO
        float packedO[8 * MAX_BLOCKN] = {0};
#endif
        int bM, bK, bN;
        for (bM = 0; bM < blockM - tileMSize + 1; bM += tileMSize) {
            for (bK = 0; bK < blockK - tileKSize + 1; bK += tileKSize) {
#if !defined(MAI_GEMM_NO_CACHE) && !defined(MAI_GEMM_CACHE_B)
                pack(aPtr + bM * K + bK, packedA, tileMSize, tileKSize, K, tileKSize);
#endif

#if !defined(MAI_GEMM_NO_CACHE)
                if (bM == 0) {
                    pack(bPtr + bK * N, packedB + bK * blockN, tileKSize, blockN, N, tileNSize);
                }
#endif

                for (bN = 0; bN < blockN - tileNSize + 1; bN += tileNSize) {
#if defined(MAI_GEMM_NO_CACHE)
                    gemm_tile_884(aPtr + bM * K + bK,
                            bPtr + bK * N + bN,
                            oPtr + bM * N + bN, M, N, K);
#elif defined(MAI_GEMM_CACHE_B)
                    gemm_tile_884_b_morden_84(aPtr + bM * K + bK,
                            packedB + bK * blockN + bN * tileKSize,
                            oPtr + bM * N + bN, M, N, K);
#elif defined(MAI_GEMM_CACHE_ABO)
                    gemm_tile_884_a_morden_88_b_morden_84_o_sequential(packedA,
                            packedB + bK * blockN + bN * tileKSize,
                            packedO + bN * tileMSize, M, N, K);
#else
                    gemm_tile_884_a_morden_88_b_morden_84(packedA,
                            packedB + bK * blockN + bN * tileKSize,
                            oPtr + bM * N + bN, M, N, K);
#endif
                }
                if (bN < blockN) {
                    gemm_random_block(aPtr + bM * K + bK, bPtr + bK * N + bN, oPtr + bM * N + bN,
                            tileMSize, blockN - bN, tileKSize, M, N, K);
                }
            }
#if defined(MAI_GEMM_CACHE_ABO)
            //write output cache into oPtr
            for (int i = 0; i < blockN - tileNSize + 1; i += tileNSize) {
                for (int j = 0; j < tileMSize; ++j) {
                    float32x4_t oValue = vld1q_f32(oPtr +  bM * N + j * N + i);
                    float32x4_t packedValue = vld1q_f32(packedO + i * tileMSize + j * tileNSize);
                    oValue = vaddq_f32(oValue, packedValue);
                    vst1q_f32(oPtr +  bM * N + j * N + i, oValue);
                }
            }
            memset(packedO, 0, sizeof(float) * 8 * MAX_BLOCKN);
#endif

            if (bK < blockK) {
                gemm_random_block(aPtr + bM * K + bK, bPtr + bK * N, oPtr + bM * N,
                        tileMSize, blockN, blockK - bK, M, N, K);
            }
        }

        if (bM < blockM) {//remaining
            gemm_random_block(aPtr + bM * K, bPtr, oPtr + bM * N,
                    blockM - bM, blockN, blockK, M, N, K);
        }
    }

    static void gemm(const float* aPtr, const float* bPtr, const float* cPtr, float* oPtr,
            const int M, const int N, const int K) {
        //ALOGI("Neon float Gemm2");
        const int tileSize = 64;
        const int mBlockSize = M / tileSize + (M % tileSize == 0 ? 0 : 1);
        const int nBlockSize = N / tileSize + (N % tileSize == 0 ? 0 : 1);
        const int kBlockSize = K / tileSize + (K % tileSize == 0 ? 0 : 1);
        int remainSize[3] = {M % tileSize, N % tileSize, K % tileSize};
        memset(oPtr, 0, M * N * sizeof(float));
#pragma omp parallel for schedule(dynamic) collapse(2)
        for (int m = 0; m < mBlockSize; m++) {
            for (int n = 0; n < nBlockSize; n++) {
                for (int k = 0; k < kBlockSize; k++) {
                    int mTileSize = (m == mBlockSize - 1 && remainSize[0] > 0) ? remainSize[0] : tileSize;
                    int nTileSize = (n == nBlockSize - 1 && remainSize[1] > 0) ? remainSize[1] : tileSize;
                    int kTileSize = (k == kBlockSize - 1 && remainSize[2] > 0) ? remainSize[2] : tileSize;
                    gemm_tile<tileSize, tileSize, tileSize>(aPtr + m * tileSize * K + k * tileSize,
                            bPtr + k * tileSize * N + n * tileSize,
                            oPtr + m * tileSize * N + n * tileSize,
                            mTileSize, nTileSize, kTileSize, M, N, K);
                }
            }
        }
    }

    static void gemm_tile_884_a_morden_88_b_morden_84(const float* aPtr, const float* bPtr, float* oPtr, int M, int N, int K) {
        float32x4_t o0Value = vld1q_f32(oPtr);
        float32x4_t o1Value = vld1q_f32(oPtr + N);
        float32x4_t o2Value = vld1q_f32(oPtr + N * 2);
        float32x4_t o3Value = vld1q_f32(oPtr + N * 3);
        float32x4_t o4Value = vld1q_f32(oPtr + N * 4);
        float32x4_t o5Value = vld1q_f32(oPtr + N * 5);
        float32x4_t o6Value = vld1q_f32(oPtr + N * 6);
        float32x4_t o7Value = vld1q_f32(oPtr + N * 7);

        float32x4_t a0Value = vld1q_f32(aPtr);
        float32x4_t a1Value = vld1q_f32(aPtr + 4);
        float32x4_t a2Value = vld1q_f32(aPtr + 4 * 2);
        float32x4_t a3Value = vld1q_f32(aPtr + 4 * 3);
        float32x4_t a4Value = vld1q_f32(aPtr + 4 * 4);
        float32x4_t a5Value = vld1q_f32(aPtr + 4 * 5);
        float32x4_t a6Value = vld1q_f32(aPtr + 4 * 6);
        float32x4_t a7Value = vld1q_f32(aPtr + 4 * 7);
        float32x4_t a8Value = vld1q_f32(aPtr + 4 * 8);
        float32x4_t a9Value = vld1q_f32(aPtr + 4 * 9);
        float32x4_t a10Value = vld1q_f32(aPtr + 4 * 10);
        float32x4_t a11Value = vld1q_f32(aPtr + 4 * 11);
        float32x4_t a12Value = vld1q_f32(aPtr + 4 * 12);
        float32x4_t a13Value = vld1q_f32(aPtr + 4 * 13);
        float32x4_t a14Value = vld1q_f32(aPtr + 4 * 14);
        float32x4_t a15Value = vld1q_f32(aPtr + 4 * 15);

        float32x4_t b0Value = vld1q_f32(bPtr);
        float32x4_t b1Value = vld1q_f32(bPtr + 4);
        float32x4_t b2Value = vld1q_f32(bPtr + 4 * 2);
        float32x4_t b3Value = vld1q_f32(bPtr + 4 * 3);
        float32x4_t b4Value = vld1q_f32(bPtr + 4 * 4);
        float32x4_t b5Value = vld1q_f32(bPtr + 4 * 5);
        float32x4_t b6Value = vld1q_f32(bPtr + 4 * 6);
        float32x4_t b7Value = vld1q_f32(bPtr + 4 * 7);

        MAI_VFMAQ_LANEQ_F32_8(o0Value, a0Value, a1Value);
        MAI_VFMAQ_LANEQ_F32_8(o1Value, a2Value, a3Value);
        MAI_VFMAQ_LANEQ_F32_8(o2Value, a4Value, a5Value);
        MAI_VFMAQ_LANEQ_F32_8(o3Value, a6Value, a7Value);
        MAI_VFMAQ_LANEQ_F32_8(o4Value, a8Value, a9Value);
        MAI_VFMAQ_LANEQ_F32_8(o5Value, a10Value, a11Value);
        MAI_VFMAQ_LANEQ_F32_8(o6Value, a12Value, a13Value);
        MAI_VFMAQ_LANEQ_F32_8(o7Value, a14Value, a15Value);

        vst1q_f32(oPtr, o0Value);
        vst1q_f32(oPtr + N, o1Value);
        vst1q_f32(oPtr + N * 2, o2Value);
        vst1q_f32(oPtr + N * 3, o3Value);
        vst1q_f32(oPtr + N * 4, o4Value);
        vst1q_f32(oPtr + N * 5, o5Value);
        vst1q_f32(oPtr + N * 6, o6Value);
        vst1q_f32(oPtr + N * 7, o7Value);
    }

    static void gemm_tile_884_a_morden_88_b_morden_84_o_sequential(const float* aPtr, const float* bPtr, float* oPtr, int M, int N, int K) {
        float32x4_t o0Value = vld1q_f32(oPtr);
        float32x4_t o1Value = vld1q_f32(oPtr + 4);
        float32x4_t o2Value = vld1q_f32(oPtr + 4 * 2);
        float32x4_t o3Value = vld1q_f32(oPtr + 4 * 3);
        float32x4_t o4Value = vld1q_f32(oPtr + 4 * 4);
        float32x4_t o5Value = vld1q_f32(oPtr + 4 * 5);
        float32x4_t o6Value = vld1q_f32(oPtr + 4 * 6);
        float32x4_t o7Value = vld1q_f32(oPtr + 4 * 7);

        float32x4_t a0Value = vld1q_f32(aPtr);
        float32x4_t a1Value = vld1q_f32(aPtr + 4);
        float32x4_t a2Value = vld1q_f32(aPtr + 4 * 2);
        float32x4_t a3Value = vld1q_f32(aPtr + 4 * 3);
        float32x4_t a4Value = vld1q_f32(aPtr + 4 * 4);
        float32x4_t a5Value = vld1q_f32(aPtr + 4 * 5);
        float32x4_t a6Value = vld1q_f32(aPtr + 4 * 6);
        float32x4_t a7Value = vld1q_f32(aPtr + 4 * 7);
        float32x4_t a8Value = vld1q_f32(aPtr + 4 * 8);
        float32x4_t a9Value = vld1q_f32(aPtr + 4 * 9);
        float32x4_t a10Value = vld1q_f32(aPtr + 4 * 10);
        float32x4_t a11Value = vld1q_f32(aPtr + 4 * 11);
        float32x4_t a12Value = vld1q_f32(aPtr + 4 * 12);
        float32x4_t a13Value = vld1q_f32(aPtr + 4 * 13);
        float32x4_t a14Value = vld1q_f32(aPtr + 4 * 14);
        float32x4_t a15Value = vld1q_f32(aPtr + 4 * 15);

        float32x4_t b0Value = vld1q_f32(bPtr);
        float32x4_t b1Value = vld1q_f32(bPtr + 4);
        float32x4_t b2Value = vld1q_f32(bPtr + 4 * 2);
        float32x4_t b3Value = vld1q_f32(bPtr + 4 * 3);
        float32x4_t b4Value = vld1q_f32(bPtr + 4 * 4);
        float32x4_t b5Value = vld1q_f32(bPtr + 4 * 5);
        float32x4_t b6Value = vld1q_f32(bPtr + 4 * 6);
        float32x4_t b7Value = vld1q_f32(bPtr + 4 * 7);

        MAI_VFMAQ_LANEQ_F32_8(o0Value, a0Value, a1Value);
        MAI_VFMAQ_LANEQ_F32_8(o1Value, a2Value, a3Value);
        MAI_VFMAQ_LANEQ_F32_8(o2Value, a4Value, a5Value);
        MAI_VFMAQ_LANEQ_F32_8(o3Value, a6Value, a7Value);
        MAI_VFMAQ_LANEQ_F32_8(o4Value, a8Value, a9Value);
        MAI_VFMAQ_LANEQ_F32_8(o5Value, a10Value, a11Value);
        MAI_VFMAQ_LANEQ_F32_8(o6Value, a12Value, a13Value);
        MAI_VFMAQ_LANEQ_F32_8(o7Value, a14Value, a15Value);

        vst1q_f32(oPtr, o0Value);
        vst1q_f32(oPtr + 4, o1Value);
        vst1q_f32(oPtr + 4 * 2, o2Value);
        vst1q_f32(oPtr + 4 * 3, o3Value);
        vst1q_f32(oPtr + 4 * 4, o4Value);
        vst1q_f32(oPtr + 4 * 5, o5Value);
        vst1q_f32(oPtr + 4 * 6, o6Value);
        vst1q_f32(oPtr + 4 * 7, o7Value);
    }

    static void gemm_tile_884_b_morden_84(const float* aPtr, const float* bPtr, float* oPtr, int M, int N, int K) {
        float32x4_t o0Value = vld1q_f32(oPtr);
        float32x4_t o1Value = vld1q_f32(oPtr + N);
        float32x4_t o2Value = vld1q_f32(oPtr + N * 2);
        float32x4_t o3Value = vld1q_f32(oPtr + N * 3);
        float32x4_t o4Value = vld1q_f32(oPtr + N * 4);
        float32x4_t o5Value = vld1q_f32(oPtr + N * 5);
        float32x4_t o6Value = vld1q_f32(oPtr + N * 6);
        float32x4_t o7Value = vld1q_f32(oPtr + N * 7);

        float32x4_t a0Value = vld1q_f32(aPtr);
        float32x4_t a1Value = vld1q_f32(aPtr + 4);
        float32x4_t a2Value = vld1q_f32(aPtr + K);
        float32x4_t a3Value = vld1q_f32(aPtr + K + 4);
        float32x4_t a4Value = vld1q_f32(aPtr + K * 2);
        float32x4_t a5Value = vld1q_f32(aPtr + K * 2 + 4);
        float32x4_t a6Value = vld1q_f32(aPtr + K * 3);
        float32x4_t a7Value = vld1q_f32(aPtr + K * 3 + 4);
        float32x4_t a8Value = vld1q_f32(aPtr + K * 4);
        float32x4_t a9Value = vld1q_f32(aPtr + K * 4 + 4);
        float32x4_t a10Value = vld1q_f32(aPtr + K * 5);
        float32x4_t a11Value = vld1q_f32(aPtr + K * 5 + 4);
        float32x4_t a12Value = vld1q_f32(aPtr + K * 6);
        float32x4_t a13Value = vld1q_f32(aPtr + K * 6 + 4);
        float32x4_t a14Value = vld1q_f32(aPtr + K * 7);
        float32x4_t a15Value = vld1q_f32(aPtr + K * 7 + 4);

        float32x4_t b0Value = vld1q_f32(bPtr);
        float32x4_t b1Value = vld1q_f32(bPtr + 4);
        float32x4_t b2Value = vld1q_f32(bPtr + 4 * 2);
        float32x4_t b3Value = vld1q_f32(bPtr + 4 * 3);
        float32x4_t b4Value = vld1q_f32(bPtr + 4 * 4);
        float32x4_t b5Value = vld1q_f32(bPtr + 4 * 5);
        float32x4_t b6Value = vld1q_f32(bPtr + 4 * 6);
        float32x4_t b7Value = vld1q_f32(bPtr + 4 * 7);

        MAI_VFMAQ_LANEQ_F32_8(o0Value, a0Value, a1Value);
        MAI_VFMAQ_LANEQ_F32_8(o1Value, a2Value, a3Value);
        MAI_VFMAQ_LANEQ_F32_8(o2Value, a4Value, a5Value);
        MAI_VFMAQ_LANEQ_F32_8(o3Value, a6Value, a7Value);
        MAI_VFMAQ_LANEQ_F32_8(o4Value, a8Value, a9Value);
        MAI_VFMAQ_LANEQ_F32_8(o5Value, a10Value, a11Value);
        MAI_VFMAQ_LANEQ_F32_8(o6Value, a12Value, a13Value);
        MAI_VFMAQ_LANEQ_F32_8(o7Value, a14Value, a15Value);

        vst1q_f32(oPtr, o0Value);
        vst1q_f32(oPtr + N, o1Value);
        vst1q_f32(oPtr + N * 2, o2Value);
        vst1q_f32(oPtr + N * 3, o3Value);
        vst1q_f32(oPtr + N * 4, o4Value);
        vst1q_f32(oPtr + N * 5, o5Value);
        vst1q_f32(oPtr + N * 6, o6Value);
        vst1q_f32(oPtr + N * 7, o7Value);
    }

    static void gemm_tile_b_morden(const float* aPtr, const float* bPtr, float* oPtr,
            int blockM, int blockN, int blockK, int M, int N, int K) {
        const int tileMSize = 8;
        const int tileKSize = 8;
        const int tileNSize = 4;
        //#pragma omp parallel for schedule(dynamic)
        for (int bM = 0; bM < blockM; bM += tileMSize) {
            for (int bK = 0; bK < blockK; bK += tileKSize) {
                for (int bN = 0; bN < blockN; bN += tileNSize) {
                    ALOGI("bM=%d, bK=%d, bN=%d, aPtr Offset=%d, bPtr offset=%d", bM, bK, bN, bM * K + bK, bK * N + bN * tileKSize);
                    gemm_tile_884_b_morden_84(aPtr + bM * K + bK,
                            bPtr + bK * N + bN * tileKSize,
                            oPtr + bM * N + bN,
                            M, N, K);
                }
            }
        }
    }

    static void gemm_b_morden(const float* aPtr, const float* bPtr, const float* cPtr, float* oPtr, int M, int N, int K) {
        ALOGI("Neon float Gemm morden");
        const int tileSize = 64;
        const int mBlockSize = M / tileSize + (M % tileSize == 0 ? 0 : 1);
        const int nBlockSize = N / tileSize + (N % tileSize == 0 ? 0 : 1);
        const int kBlockSize = K / tileSize + (K % tileSize == 0 ? 0 : 1);
        //ALOGI("mBlockSize=%d, nBlockSize=%d, kBlockSize=%d", mBlockSize, nBlockSize, kBlockSize);
        int remainSize[3] = {M % tileSize, N % tileSize, K % tileSize};
        memset(oPtr, 0, M * N * sizeof(float));
        //#pragma omp parallel for schedule(dynamic)
#pragma omp parallel for schedule(dynamic) collapse(2)
        for (int m = 0; m < mBlockSize; m++) {
            //memcpy(oPtr + m * N, cPtr, N * sizeof(float));
            //memcpy(oPtr + (m + 1) * N, cPtr, N * sizeof(float));
            //memcpy(oPtr + (m + 2) * N, cPtr, N * sizeof(float));
            //memcpy(oPtr + (m + 3) * N, cPtr, N * sizeof(float));
            for (int n = 0; n < nBlockSize; n++) {
                for (int k = 0; k < kBlockSize; k++) {
                    int mTileSize = (m == mBlockSize - 1 && remainSize[0] > 0) ? remainSize[0] : tileSize;
                    int nTileSize = (n == nBlockSize - 1 && remainSize[1] > 0) ? remainSize[1] : tileSize;
                    int kTileSize = (k == kBlockSize - 1 && remainSize[2] > 0) ? remainSize[2] : tileSize;
                    ALOGI("m=%d, n=%d, k=%d, mTileSize=%d, nTileSize=%d, kTileSize=%d", m, n, k, mTileSize, nTileSize, kTileSize);
                    gemm_tile_b_morden(aPtr + m * tileSize * K + k * tileSize,
                            bPtr + k * tileSize * N + n * tileSize * 8,
                            oPtr + m * tileSize * N + n * tileSize,
                            mTileSize, nTileSize, kTileSize, M, N, K);
                }
            }
        }
    }

};

} // namespace NEON
} // namespace CPU
} // namespace Op
} // namespace MAI
