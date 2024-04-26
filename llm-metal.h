#pragma once
//
// LLM routines for Apple Metal
// 2024.4
//
// A set of simple routines for running LLMs on Apple Silicon GPU using Metal.
// The functions wrap around Metal so that callers do not need to call Metal API
// directly. These do 3 things,
//
// 1. Metal buffer management: lm_alloc() and lm_free()
//
// 2. Operator command queueing: lm_add(), lm_gemv(), lm_rmsnorm(), ...
//    Calling these enqueues GPU operations on the buffers allocated by lm_alloc().
//    Building up a sizable command queue is necessary to achieve good GPU performance
//    as it amortizes the launching costs of GPU commands.
//
// 3. Operator command execution: op_execute(). It runs and empties the command queue.
//

// Call this before everthing else.
// return 0 on success, -1 on failure
int lm_init(void);

// Allocate page-aligned memory and register a Metal buffer around it
// return 0 if successful
int lm_alloc(float **buf, unsigned long long size);

// Free a buffer allocated with lm_alloc()
void lm_free(float *p);

// The following only add the operator into the command queue.
// Actual computation happens when op_execute() is called.
// Offsets (xoff, yoff, woff) are in number of floats.

// Vector add
// y = x + w
void lm_add(float *y, float *x, float *w, int n, int yoff, int xoff, int woff);

// Matrix and vector multiplication
// y = W * x
void lm_gemv(float *y, float *x, float *w, int n, int d, int yoff, int woff);

// RMS normalization
void lm_rmsnorm(float *y, float *x, float *w, int n, float eps, int woff);

// SwiGLU non-linearity
void lm_swiglu(float *y, float *x, float *x2, int n);

// RoPE - Rotary Positional Embedding
// qwen: 1 if qwen version of rope, 0 if not
void lm_rope(int qwen, float *q, float *k, int pos, int nq, int nk, int head_size, float theta, int koff);

// The following 3 together are the multihead attention.
// d - total rows of key to process
void lm_multihead_attention(float *att, float *q, float *k, int head_size, int n_heads, int d, int koff);

// y = softmax(x)
// x: (d x n) matrix (softmax along rows)
void lm_softmax(float *y, float *x, int n, int d);

// scale each row segment (head) of a v[t] by att[h][t], then add the rows together to get y
// v: value matrix (d x n_heads x head_size)
// att: attention matrix (n_heads x d)
// each thread owns a column of the value matrix
void lm_multihead_weighted_sum(float *y, float *att, float *v, int head_size, int n_heads, int d, int voff);

// execute all enqueued operations and empties the operator queue
void lm_execute();