#include "core/GemmTest.h"
namespace MAI {
namespace Test {

class Gemm1Test : public GemmTest {
public:
    Gemm1Test() {
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

TEST_F(Gemm1Test, basic) {
    long start = nowMicros();
    gemm_random(A, B, C, M, N, K);
    long end = nowMicros();
    printf("Gemm1Test cost:%d ms\n", (end - start)/ 1000);
}

} // namespace Test
} // namespace MAI
