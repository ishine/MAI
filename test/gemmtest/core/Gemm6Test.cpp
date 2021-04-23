#include "core/GemmTest.h"
namespace MAI {
namespace Test {

class Gemm6Test : public GemmTest {
public:
    Gemm6Test() {
        //int aIndex = 0;
        //int bIndex = 0;
        A = randomValue(M * K/*, [&aIndex]() -> float {return ++aIndex;}*/);
        B = randomValue(K * N/*, [&bIndex]() -> float {return ++bIndex;}*/);
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

TEST_F(Gemm6Test, checkResult) {
    float* checkC = new float[M * N];
    gemm_random(A, B, checkC, M, N, K);
    gemm_op6(A, B, C, M, N, K);
    //ExpectEQ(C, checkC, M * N);
}

TEST_F(Gemm6Test, basic) {
    long start = nowMicros();
    gemm_op6(A, B, C, M, N, K);
    long end = nowMicros();
    printf("Gemm6Test cost:%d ms\n", (end - start)/ 1000);
}

} // namespace Test
} // namespace MAI
