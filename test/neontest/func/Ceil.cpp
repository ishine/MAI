#include <arm_neon.h>
#include <cmath>
#include "core/TestUtil.h"
/**
 * void ceil(float x) {
       int n = (int)x;
       float r = (float)n;
       r = r + (r < x);
   }
 */
void ceil(float* dst, float* src, unsigned int count) {
    for (unsigned int i = 0; i < (count & ~2); i += 4) {
        float32x4_t x = vld1q_f32(src + i);
        float32x4_t r = vcvtq_f32_s32(vcvtq_s32_f32(x));
        uint32x4_t b = vshrq_n_u32(vcltq_f32(r, x), 31);
        r = vaddq_f32(r, vcvtq_f32_u32(b));
        vst1q_f32(dst + i, r);
    }
    for (unsigned int i = (count & ~2); i < count; ++i) {
        dst[i] = std::ceil(src[i]);
    }
}
