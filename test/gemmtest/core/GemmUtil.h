#pragma once

#include <random>
#include <sys/time.h>

namespace MAI {
namespace Test {

void gemm_random(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_op2(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_op3(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_op4(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_op5(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_block(const float* A, const float* B, float* C, int MB, int NB, int KB,
        int M, int N, int K);
void gemm_tile_884(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_op6(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_block_neon(const float* A, const float* B, float* C, int MB, int NB, int KB,
        int M, int N, int K);
void gemm_tile_884_neon(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_op7(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_op8(const float* A, const float* B, float* C, int M, int N, int K);


inline float* randomValue(int size, std::function<float()> func
        = []() -> float {
            return (static_cast<float>(random()) / RAND_MAX - 0.5f);
        }) {
    float* data = new float[size];
    for (int i = 0; i < size; ++i) {
        data[i] = func();
    }
    return data;
}

inline float* fillValue(int size, float value) {
    return randomValue(size, [&value]()->float{return value;});
}

inline uint64_t nowMicros() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
}

} // namespace Test
} // namespace MAI
