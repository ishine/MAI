#include "core/GemmTest.h"
#include "core/SGemmUtil.h"
namespace MAI {
namespace Test {

class SGemmOpTest : public GemmTest {
public:
    SGemmOpTest() {
        //int aIndex = 0;
        //int bIndex = 0;
        A = randomValue(M * K/*, [&aIndex]() -> float {return ++aIndex;}*/);
        B = randomValue(K * N/*, [&bIndex]() -> float {return ++bIndex;}*/);
        C = new float[M * N];
        zeros(C, M * N);
    }

    void zeros(float* v, int size) {
        memset(v, 0, size * sizeof(float));
    }

    void printOutput(const float* v, int size, const std::string& prefix) {
        printf("%s=========================>\n", prefix.c_str());
        for (int i = 0; i < size; ++i) {
            printf("%s[%d]: %f\n", prefix.c_str(), i, v[i]);
        }
    }

protected:
    float* A;
    float* B;
    float* C;
    int M{1024};
    int K{1024};
    int N{1024};
};

TEST_F(SGemmOpTest, r_tn_checkResult) {
    float* checkC = new float[M * N];
    zeros(checkC, M * N);
    sgemm(true, true, false,
            M, N, K,
            1.f, A, M,
            B, N,
            1.f, checkC, N);
    sgemm_op(true, true, false,
            M, N, K,
            1.f, A, M,
            B, N,
            1.f, C, N);
    ExpectEQ(C, checkC, M * N);
}

TEST_F(SGemmOpTest, r_tn) {
    long start = nowMicros();
    sgemm_op(true, true, false,
            M, N, K,
            1.f, A, M,
            B, N,
            1.f, C, N);
    long end = nowMicros();
    printf("SGemmOpTest r_tn cost:%d ms\n", (end - start)/ 1000);
}

TEST_F(SGemmOpTest, r_nt_checkResult) {
    float* checkC = new float[M * N];
    zeros(checkC, M * N);
    sgemm(true, false, true,
            M, N, K,
            1.f, A, K,
            B, K,
            1.f, checkC, N);
    sgemm_op(true, false, true,
            M, N, K,
            1.f, A, K,
            B, K,
            1.f, C, N);
    ExpectEQ(C, checkC, M * N);
}

TEST_F(SGemmOpTest, r_nt) {
    long start = nowMicros();
    sgemm_op(true, false, true,
            M, N, K,
            1.f, A, K,
            B, K,
            1.f, C, N);
    long end = nowMicros();
    printf("SGemmOpTest r_nt cost:%d ms\n", (end - start)/ 1000);
}

} // namespace Test
} // namespace MAI
