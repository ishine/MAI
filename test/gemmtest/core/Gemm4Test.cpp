#include "core/GemmTest.h"
namespace MAI {
namespace Test {

class Gemm4Test : public GemmTest {
public:
    Gemm4Test() {
        A = randomValue(M * K);
        B = randomValue(K * N);
        C = new float[M * N];
    }
protected:
    float* A;
    float* B;
    float* C;
    int M{1024};
    int K{1024};
    int N{1024};
};

TEST_F(Gemm4Test, checkResult) {
    float* checkC = new float[M * N];
    gemm_random(A, B, checkC, M, N, K);
    gemm_op4(A, B, C, M, N, K);
    bool checkResult = true;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (C[i * N + j] != checkC[i * N + j]) {
                printf("[%d, %d] check failed, real:%f, expected:%f\n",
                    i, j, C[i * N + j], checkC[i * N + j]);
                checkResult = false;
            }
        }
    }
    EXPECT_TRUE(checkResult);
}

TEST_F(Gemm4Test, basic) {
    long start = nowMicros();
    gemm_op4(A, B, C, M, N, K);
    long end = nowMicros();
    printf("Gemm4Test cost:%d ms\n", (end - start)/ 1000);
}

} // namespace Test
} // namespace MAI
