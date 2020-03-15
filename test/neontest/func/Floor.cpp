#include <arm_neon.h>
#include <cmath>
#include "core/TestUtil.h"

float floor_c(float x) {
       int n = (int)x;
       float r = (float)n;
       r = r - (r > x);
}
/**
 * void floor(float x) {
       int n = (int)x;
       float r = (float)n;
       r = r - (r > x);
   }
 */
void floor(float* dst, float* src, unsigned int count) {
    for (unsigned int i = 0; i < (count & ~2); i += 4) {
        float32x4_t x = vld1q_f32(src + i);
        float32x4_t r = vcvtq_f32_s32(vcvtq_s32_f32(x));
        uint32x4_t b = vshrq_n_u32(vcgtq_f32(r, x), 31);
        r = vsubq_f32(r, vcvtq_f32_u32(b));
        vst1q_f32(dst + i, r);
    }
    for (unsigned int i = (count & ~2); i < count; ++i) {
        dst[i] = std::floor(src[i]);
    }
}
