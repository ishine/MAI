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

#include "include/Type.h"

namespace MAI {
namespace Op {
namespace CPU {
namespace Ref {

template<typename T, bool transpose_a, bool transpose_b>
struct Gemm {
    // O = C + A * B;
    // A: MxK B: KxN C: MxN
    static void gemm(const T* aPtr, const T* bPtr, const T* cPtr, T* oPtr, int M, int N, int K) {
        ALOGI("Ref::Gemm");
        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < K; ++k) {
                for (int n = 0; n < N; ++n) {
                //oPtr[m * N + n] = cPtr[n];
                //oPtr[m * N + n] = 0;
                    oPtr[m * N + n] += aPtr[m * K + k] * bPtr[k * N + n];
                }
            }
        }
    }
};

} // namespace Ref
} // namespace CPU
} // namespace Op
} // namespace MAI
