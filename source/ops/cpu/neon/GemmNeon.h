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
#include "omp.h"
#include "include/Type.h"

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
struct Gemm {
    static void addDot1x4a(const T* aPtr, const T* bPtr, T* oPtr, int K, int N) {
        register T o0Reg, o1Reg, o2Reg, o3Reg, a0Reg;
        o0Reg = 0.f;
        o1Reg = 0.f;
        o2Reg = 0.f;
        o3Reg = 0.f;
        const T* b0Ptr = &bPtr[0];
        const T* b1Ptr = &bPtr[1];
        const T* b2Ptr = &bPtr[2];
        const T* b3Ptr = &bPtr[3];
        for (int k = 0; k < K; k += 4) {
            a0Reg = aPtr[k];

            o0Reg += a0Reg * *b0Ptr;
            o1Reg += a0Reg * *b1Ptr;
            o2Reg += a0Reg * *b2Ptr;
            o3Reg += a0Reg * *b3Ptr;

            b0Ptr += N;
            b1Ptr += N;
            b2Ptr += N;
            b3Ptr += N;

            a0Reg = aPtr[k + 1];

            o0Reg += a0Reg * *b0Ptr;
            o1Reg += a0Reg * *b1Ptr;
            o2Reg += a0Reg * *b2Ptr;
            o3Reg += a0Reg * *b3Ptr;

            b0Ptr += N;
            b1Ptr += N;
            b2Ptr += N;
            b3Ptr += N;

            a0Reg = aPtr[k + 2];

            o0Reg += a0Reg * *b0Ptr;
            o1Reg += a0Reg * *b1Ptr;
            o2Reg += a0Reg * *b2Ptr;
            o3Reg += a0Reg * *b3Ptr;

            b0Ptr += N;
            b1Ptr += N;
            b2Ptr += N;
            b3Ptr += N;

            a0Reg = aPtr[k + 3];

            o0Reg += a0Reg * *b0Ptr;
            o1Reg += a0Reg * *b1Ptr;
            o2Reg += a0Reg * *b2Ptr;
            o3Reg += a0Reg * *b3Ptr;

            b0Ptr += N;
            b1Ptr += N;
            b2Ptr += N;
            b3Ptr += N;
        }
        *oPtr += o0Reg;
        *(oPtr + 1) += o1Reg;
        *(oPtr + 2) += o2Reg;
        *(oPtr + 3) += o3Reg;
    }

    // O = C + A * B;
    // A: MxK B: KxN C: MxN
    static void gemma(const T* aPtr, const T* bPtr, const T* cPtr, T* oPtr, int M, int N, int K) {
        ALOGI("Neon Gemm");
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; n += 4) {
                oPtr[m * N + n] = cPtr[n];
                oPtr[m * N + n + 1] = cPtr[n + 1];
                oPtr[m * N + n + 2] = cPtr[n + 2];
                oPtr[m * N + n + 3] = cPtr[n + 3];
                addDot1x4(&aPtr[m * K], &bPtr[n], &oPtr[m * N + n], K, N);

            }
        }
    }

    static void addDot1x4(const T* aPtr, const T* bPtr, T* toPtr, int N) {
        T* oPtr = toPtr;

        const T* a0Ptr = aPtr;
        const T* a1Ptr = aPtr + 1;
        const T* a2Ptr = aPtr + 2;
        const T* a3Ptr = aPtr + 3;

        const T* b0Ptr = bPtr;
        const T* b1Ptr = bPtr + N;
        const T* b2Ptr = bPtr + N * 2;
        const T* b3Ptr = bPtr + N * 3;

        for (int n = 0; n < N; n += 4) {
            *oPtr += *a0Ptr * *b0Ptr;
            *oPtr += *a1Ptr * *b1Ptr;
            *oPtr += *a2Ptr * *b2Ptr;
            *oPtr += *a3Ptr * *b3Ptr;
            oPtr++;

            *oPtr += *a0Ptr * *(b0Ptr + 1);
            *oPtr += *a1Ptr * *(b1Ptr + 1);
            *oPtr += *a2Ptr * *(b2Ptr + 1);
            *oPtr += *a3Ptr * *(b3Ptr + 1);
            oPtr++;

            *oPtr += *a0Ptr * *(b0Ptr + 2);
            *oPtr += *a1Ptr * *(b1Ptr + 2);
            *oPtr += *a2Ptr * *(b2Ptr + 2);
            *oPtr += *a3Ptr * *(b3Ptr + 2);
            oPtr++;

            *oPtr += *a0Ptr * *(b0Ptr + 3);
            *oPtr += *a1Ptr * *(b1Ptr + 3);
            *oPtr += *a2Ptr * *(b2Ptr + 3);
            *oPtr += *a3Ptr * *(b3Ptr + 3);
            oPtr++;
            b0Ptr += 4;
            b1Ptr += 4;
            b2Ptr += 4;
            b3Ptr += 4;
        }
    }

    // O = C + A * B;
    // A: MxK B: KxN C: MxN
    static void gemm(const T* aPtr, const T* bPtr, const T* cPtr, T* oPtr, int M, int N, int K) {
        ALOGI("Neon Gemm");
        for (int m = 0; m < M; ++m) {
            memcpy(oPtr + m * N, cPtr, N * sizeof(T));
            for (int k = 0; k < K; k += 4) {
                addDot1x4(&aPtr[m * K + k], &bPtr[k * N], &oPtr[m * N], N);
                //addDot1x4(&aPtr[m * K + k + 1], &bPtr[(k + 1) * N], &oPtr[m * N], N);
                //addDot1x4(&aPtr[m * K + k + 2], &bPtr[(k + 2) * N], &oPtr[m * N], N);
                //addDot1x4(&aPtr[m * K + k + 3], &bPtr[(k + 3) * N], &oPtr[m * N], N);
            }
        }
    }
};

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

    static void pack(const float* bPtr, float* packedBPtr, int blockM, int blockN, int N, int tile) {
        //ALOGI("packB M=%d, N=%d, tile=%d", blockM, N, tile);
        for(int i = 0; i < blockN; i += tile) {
            for (int j = 0; j < blockM; ++j) {
                for (int t = 0; t < tile; ++t) {
                    *packedBPtr++ = *(bPtr + j * N + i + t);
                }
            }
        }
        //ALOGI("PackB end");
    }

    static void gemm_tile(const float* aPtr, const float* bPtr, float* oPtr,
            int blockM, int blockN, int blockK, int M, int N, int K) {
        float packedA[blockM * blockN];
        float packedB[blockK * blockN];
        int bM, bK, bN;
        const int tileMSize = 8;
        const int tileKSize = 8;
        const int tileNSize = 4;
        //#pragma omp parallel for schedule(dynamic)
        for (bM = 0; bM < blockM - tileMSize + 1; bM += tileMSize) {
            for (bK = 0; bK < blockK - tileKSize + 1; bK += tileKSize) {
                pack(aPtr + bM * blockK, packedB + bM * blockK, tileMSize, tileMSize, K, tileKSize);
                if (bM == 0) {
                    pack(bPtr + bK * blockN, packedB + bK * blockN, tileKSize, blockN, N, tileNSize);
                }
                for (bN = 0; bN < blockN - tileNSize + 1; bN += tileNSize) {
                    //ALOGI("bM=%d, bN=%d, bK=%d", bM, bN, bK);
                    //gemm_tile_884(aPtr + bM * K + bK, bPtr + bK * N + bN, oPtr + bM * N + bN,
                    //        M, N, K);
                    //gemm_tile_884_b_morden_44(aPtr + bM * K + bK, packedB + bK * bN, oPtr + bM * N + bN,
                    //        M, N, K);
                    gemm_tile_884_a_morden_88_b_morden_44(aPtr + bM * K + bK, packedB + bK * bN, oPtr + bM * N + bN,
                            M, N, K);
                }
                //ALOGI("end bk=%d", bK);
                if (bN < blockN) {
                    ALOGI("bN < blockN, bN=%d, blockN=%d", bN, blockN);
                    gemm_random_block(aPtr + bM * K + bK, bPtr + bK * N + bN, oPtr + bM * N + bN,
                            tileMSize, blockN - bN, tileKSize, M, N, K);
                }
            }

            if (bK < blockK) {
                ALOGI("bK < blockK, bK=%d, blockK=%d", bK, blockK);
                gemm_random_block(aPtr + bM * K + bK, bPtr + bK * N, oPtr + bM * N,
                        tileMSize, blockN, blockK - bK, M, N, K);
            }
        }

        if (bM < blockM) {//remaining
            ALOGI("bM < blockM, bM=%d, blockM=%d", bM, blockM);
            gemm_random_block(aPtr + bM * K, bPtr, oPtr + bM * N,
                    blockM - bM, blockN, blockK, M, N, K);
        }
    }

    static void gemm_2(const float* aPtr, const float* bPtr, const float* cPtr, float* oPtr, int M, int N, int K) {
        ALOGI("Neon float Gemm2");
        const int tileSize = 64;
        const int mBlockSize = M / tileSize + (M % tileSize == 0 ? 0 : 1);
        const int nBlockSize = N / tileSize + (N % tileSize == 0 ? 0 : 1);
        const int kBlockSize = K / tileSize + (K % tileSize == 0 ? 0 : 1);
        int remainSize[3] = {M % tileSize, N % tileSize, K % tileSize};
        //ALOGI("mBlockSize=%d, nBlockSize=%d, kBlockSize=%d", mBlockSize, nBlockSize, kBlockSize);
        memset(oPtr, 0, M * N * sizeof(float));
        for (int m = 0; m < mBlockSize; m++) {
        #pragma omp parallel for schedule(dynamic)
            for (int n = 0; n < nBlockSize; n++) {
                for (int k = 0; k < kBlockSize; k++) {
                    //ALOGI("m=%d, n=%d, k=%d", m, n, k);
                    int mTileSize = (m == mBlockSize - 1 && remainSize[0] > 0) ? remainSize[0] : tileSize;
                    int nTileSize = (n == nBlockSize - 1 && remainSize[1] > 0) ? remainSize[1] : tileSize;
                    int kTileSize = (k == kBlockSize - 1 && remainSize[2] > 0) ? remainSize[2] : tileSize;
                    gemm_tile(aPtr + m * tileSize * K + k * tileSize, bPtr + k * tileSize * N + n * tileSize,
                            oPtr + m * tileSize * N + n * tileSize, mTileSize, nTileSize, kTileSize, M, N, K);
                }
            }
        }
    }

    static void gemm_tile_884_a_morden_88_b_morden_44(const float* aPtr, const float* bPtr, float* oPtr, int M, int N, int K) {
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

    static void gemm_tile_884_b_morden_44(const float* aPtr, const float* bPtr, float* oPtr, int M, int N, int K) {
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
                    gemm_tile_884_b_morden_44(aPtr + bM * K + bK, bPtr + bK * N + bN, oPtr + bM * N + bN,
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
        ALOGI("mBlockSize=%d, nBlockSize=%d, kBlockSize=%d", mBlockSize, nBlockSize, kBlockSize);
        #pragma omp parallel for schedule(dynamic)
        for (int m = 0; m < mBlockSize; m++) {
            //memcpy(oPtr + m * N, cPtr, N * sizeof(float));
            //memcpy(oPtr + (m + 1) * N, cPtr, N * sizeof(float));
            //memcpy(oPtr + (m + 2) * N, cPtr, N * sizeof(float));
            //memcpy(oPtr + (m + 3) * N, cPtr, N * sizeof(float));
            for (int n = 0; n < nBlockSize; n++) {
                for (int k = 0; k < kBlockSize; k++) {
                    gemm_tile_b_morden(aPtr + m * tileSize * K + k * tileSize, bPtr + k * tileSize * N + n * tileSize,
                            oPtr + m * tileSize * N + n * tileSize, tileSize, tileSize, tileSize, M, N, K);
                }
            }
        }
    }
};

} // namespace NEON
} // namespace CPU
} // namespace Op
} // namespace MAI
