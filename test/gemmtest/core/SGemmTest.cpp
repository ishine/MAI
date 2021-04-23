#include "core/GemmTest.h"

namespace MAI {
namespace Test {

class SGemmTest : public GemmTest {
public:
    void printOutput(const float* v, int size, const std::string& prefix) {
        printf("%s=========================>\n", prefix.c_str());
        for (int i = 0; i < size; ++i) {
            printf("%s[%d]: %f\n", prefix.c_str(), i, v[i]);
        }
    }
protected:
    float A[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float B[8] = {1,2,3,4,5,6,7,8};
    float C[8] = {0};
    int M{2};
    int K{2};
    int N{4};
};

TEST_F(SGemmTest, r_nn) {
    long start = nowMicros();
    sgemm(true, false, false,
            M, N, K,
            1.f, A, K,
            B, N,
            1.f, C, N);
    long end = nowMicros();
    printf("SGemmTest cost:%d ms\n", (end - start)/ 1000);
    printOutput(C, 4, "r_nn");
}

TEST_F(SGemmTest, r_tn) {
    long start = nowMicros();
    sgemm(true, true, false,
            M, N, K,
            1.f, A, M,
            B, N,
            1.f, C, N);
    long end = nowMicros();
    printf("SGemmTest cost:%d ms\n", (end - start)/ 1000);
    printOutput(C, 4, "r_tn");
}

TEST_F(SGemmTest, r_nt) {
    long start = nowMicros();
    sgemm(true, false, true,
            M, N, K,
            1.f, A, K,
            B, K,
            1.f, C, N);
    long end = nowMicros();
    printf("SGemmTest cost:%d ms\n", (end - start)/ 1000);
    printOutput(C, 4, "r_nt");
}

TEST_F(SGemmTest, r_tt) {
    long start = nowMicros();
    sgemm(true, true, true,
            M, N, K,
            1.f, A, M,
            B, K,
            1.f, C, N);
    long end = nowMicros();
    printf("SGemmTest cost:%d ms\n", (end - start)/ 1000);
    printOutput(C, 4, "r_tt");
}

} // namespace Test
} // namespace MAI
