        .text
        .align 4
        .global transpose

transpose:
        vld1.32 {q0}, [r1]!
        vld1.32 {q1}, [r1]!
        vld1.32 {q2}, [r1]!
        vld1.32 {q3}, [r1]!
        vtrn.32 q1 , q0
        vtrn.32 q3, q2
        vst1.32 {q0}, [r1]!
        vst1.32 {q1}, [r1]!
        vst1.32 {q2}, [r1]!
        vst1.32 {q3}, [r1]!
