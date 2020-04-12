#pragma once

#include <gtest/gtest.h>
#include <arm_neon.h>
#include "TestUtil.h"
#include "GemmUtil.h"
#include "SGemmUtil.h"

namespace MAI {
namespace Test {

class GemmTest : public testing::Test {
protected:
    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

} // namespace Test
} // namespace MAI
