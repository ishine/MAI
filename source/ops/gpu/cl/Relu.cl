__kernel void relu(__global float* input, __global float* output) {
    const int gidX = get_global_id(0);
    output[gidX] = fmax(input[gidX], 0);
}
