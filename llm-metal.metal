#include <metal_stdlib>
using namespace metal;

// https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32

/*
 * Simple matrix-vector multiplication kernel
 * y: output vector
 * w: weight matrix (row-major)
 * x: input vector (column vector)
 * n: dimension of the input vector
 *
 * This divides each row among 32 threads (a SIMD group). A threadgroup handles 32 rows.
 */
kernel void gemv (  device float *y,
                    const device float *x,
                    const device float *w,
                    const constant int& n,
                    const constant int& d,
                    uint id   [[thread_position_in_grid]],
                    uint sitg [[simdgroup_index_in_threadgroup]],
                    uint tiis [[thread_index_in_simdgroup]],
                    uint   ntg[[threads_per_threadgroup]] ) {
    // divide by N_SIMDGROUP, round up to multiples of 4
    int cols_per_thread = (n + 4*N_SIMDWIDTH - 1) / (4*N_SIMDWIDTH) * 4;
    int col = tiis * cols_per_thread;
    int row = id / N_SIMDWIDTH;
    int off = row * n;
    float4 sum = 0;
    if (col < n) {
        for (int i = col; i < col + cols_per_thread && i < n; i += 4) {
            device float4 *wp = (device float4 *)(w + off + i);
            device float4 *xp = (device float4 *)(x + i);
            sum += (*wp) * (*xp);
        }
    }
    sum = simd_sum(sum);        // collect sum across simd group
    if (tiis == 0) 
        y[row] = sum[0] + sum[1] + sum[2] + sum[3];
}

kernel void swiglu (    device float *y,
                        device float *x,
                        device float *x2,       // w3(x)
                        const constant int& n,
                        uint tpitg[[thread_position_in_grid]],
                        uint   ntg[[threads_per_threadgroup]] ) {
    for (int i = tpitg; i < n/4; i += ntg) {
        device const float4 *px = (device float4 *) (x + i * 4);
        device const float4 *px2 = (device float4 *) (x2 + i * 4);
        device float4 *py = (device float4 *) (y + i * 4);
        float4 val = *px;
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= 1.0f / (1.0f + exp(-val));
        // elementwise multiply with w3(x)
        val *= *px2;
        *py = val;
    }
}

kernel void rmsnorm (   device float         * y,
                        const device float   * x,
                        const device float   * w,
                        const constant int   & n,
                        const constant float & eps,
                        threadgroup float    * buf [[threadgroup(0)]],
                        uint tpitg[[thread_position_in_threadgroup]],
                        uint sgitg[[simdgroup_index_in_threadgroup]],
                        uint tiisg[[thread_index_in_simdgroup]],
                        uint   ntg[[threads_per_threadgroup]] ) {
    // sum of squares of x
    float sum = 0;

    for (int i = tpitg; i < n; i += ntg)    // sum all values belong to this thread
        sum += x[i] * x[i];
    
    sum = simd_sum(sum);                    // sum across simd group and broadcast

    if (ntg > N_SIMDWIDTH) {                // more than 1 simd group, then add all simd groups up
        if (sgitg == 0) 
            buf[tiisg] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0)
            buf[sgitg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    const float mean = sum / n;
    const float scale = 1.0f/sqrt(mean + eps);
    for (int i = tpitg; i < n; i += ntg)
        y[i] = w[i] * x[i] * scale;    
}

kernel void add (       device float *y,
                        device float *x,
                        device float *x2,
                        const constant int& n,
                        uint tpitg[[thread_position_in_grid]],
                        uint   ntg[[threads_per_threadgroup]] ) {
    for (int i = tpitg; i < n/4; i += ntg) {
        device const float4 *px = (device float4 *) (x + i * 4);
        device const float4 *px2 = (device float4 *) (x2 + i * 4);
        device float4 *py = (device float4 *) (y + i * 4);
        *py = (*px) + (*px2);
    }
}

// RoPE relative positional encoding
// This rotates pairs of values for each head vector in k and q
// Weight layout is different from paper. See: https://github.com/juncongmoo/pyllama/issues/83
// launch with: min(nq/2, max_threadgroup_size) threads
kernel void rope (      device float *q,
                        device float *k,
                        const constant int& pos,
                        const constant int& nq,
                        const constant int& nk,
                        const constant int& head_size,
                        const constant float& theta,
                        uint tpitg[[thread_position_in_grid]],
                        uint   ntg[[threads_per_threadgroup]] ) {
    const int halfhead = head_size / 2;
    for (int p = tpitg; p < nq / 2; p += ntg) {
        int off = p / halfhead * head_size;
        int i = p % halfhead;      // index within half of head
        float freq = 1.0f / pow(theta, (float)i / halfhead);
        float val = pos * freq;
        float fcr = cos(val);
        float fci = sin(val);
        float v0 = q[off + i];
        float v1 = q[off + halfhead + i];
        q[off + i]            = v0 * fcr - v1 * fci;
        q[off + halfhead + i] = v0 * fci + v1 * fcr;
        if (p < nk / 2) {
            float v0 = k[off + i];
            float v1 = k[off + halfhead + i];
            k[off + i]            = v0 * fcr - v1 * fci;
            k[off + halfhead + i] = v0 * fci + v1 * fcr;
        } 
    }
}

//-------------------------------------------------------------------------------------
// Multi-headed Attention kernels
// 
// calculate attention for all heads
// key: key matrix (d x n_heads x head_size)
// each threads owns a head in a row of the key matrix. so total threads = d x n_heads
kernel void multihead_attention( device float *att,
                        const device float *query,
                        const device float *key,
                        const constant int& head_size,
                        const constant int& n_heads,
                        const constant int& d,
                        uint id [[ thread_position_in_grid ]]) {
    int h = id % n_heads;
    int t = id / n_heads;
    float val = 0.0f;
    int qoff = h * head_size;
    int koff = id * head_size;
    for (int i = 0; i < head_size; i++)
        val += query[qoff + i] * key[koff + i];
    att[h * d + t] = val / sqrt((float)head_size);
}

// naive softmax kernel for d x n matrix (softmax along the rows)
// each thread owns a row of the matrix
kernel void softmax(device float *y,
                    const device float *x,
                    const constant int& n,
                    uint id [[ thread_position_in_grid ]]) {
    float m = -INFINITY;
    int off = id * n;
    for (int i = 0; i < n; i++)
        m = max(m, x[off + i]);
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
        sum += exp(x[off + i] - m);
    for (int i = 0; i < n; i++)
        y[off + i] = exp(x[off + i] - m) / sum;
}

// scale each row segment (head) of a v[t] by att[h][t], then add the rows together to get y
// v: value matrix (d x n_heads x head_size)
// att: attention matrix (n_heads x d)
// each thread owns a column of the value matrix
kernel void multihead_weighted_sum( device float *y,
                                    const device float *att,
                                    const device float *v,
                                    const constant int& head_size,
                                    const constant int& n_heads,
                                    const constant int& d,
                                    uint id [[ thread_position_in_grid ]]) {
    float val = 0.0f;
    int h = id / head_size;
    for (int t = 0; t < d; t++)
        val += v[t * n_heads * head_size + id] * att[h * d + t];
    y[id] = val;
}