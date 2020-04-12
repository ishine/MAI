#include <stdio.h>
#include <gtest/gtest.h>
#include "MAIEnvironment.h"
namespace MAI {
namespace Test {
int Main(int argc, char** argv) {
    printf("Running main() from %s\n", __FILE__);
    testing::AddGlobalTestEnvironment(new MAIEnvironment);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
} // namespace Test
} // namespace MAI
