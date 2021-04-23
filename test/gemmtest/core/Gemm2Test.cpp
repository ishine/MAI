#include "core/GemmTest.h"
namespace MAI {
namespace Test {

class Gemm2Test : public GemmTest {
public:
    Gemm2Test() {
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

TEST_F(Gemm2Test, basic) {
    long start = nowMicros();
    gemm_op2(A, B, C, M, N, K);
    long end = nowMicros();
    printf("Gemm2Test cost:%d ms\n", (end - start)/ 1000);
}

} // namespace Test
} // namespace MAI
