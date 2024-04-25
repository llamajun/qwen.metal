#import <unistd.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <sys/time.h>

#include "llm-metal.h"

// This file assumes ARC is enabled. For instance, `buffers` retain references to MTLBuffer objects.
// And rely on ARC to free them when lm_free() is called. 

static id<MTLDevice> device;
static id<MTLLibrary> library;
static id<MTLCommandQueue> commandQueue;
static id<MTLComputePipelineState> psoGemv, psoRmsnorm, psoSwiglu, psoAdd, psoRope, 
                            psoMultiheadAttention, psoSoftmax, psoMultiheadWeightedSum;
static id<MTLCommandBuffer> commandBuffer;
static id<MTLComputeCommandEncoder> encoder;


NSMutableDictionary *buffers;

int lm_init(void) {
    device = MTLCreateSystemDefaultDevice();
    commandQueue = [device newCommandQueueWithMaxCommandBufferCount:1024];
    NSString *path = @"llm-metal.metal";
    NSString *src = [NSString stringWithContentsOfFile:path
                                                encoding:NSUTF8StringEncoding
                                                    error:NULL];
    NSError *error = NULL;
    MTLCompileOptions *options = [MTLCompileOptions new];
    library = [device newLibraryWithSource:src options:options error:&error];
    if (error) {
        printf("%s: error: %s\n", __func__, [[error description] UTF8String]);
        return -1;
    }
    psoAdd = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"add"] error:&error];
    psoGemv = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gemv"] error:&error];
    psoRmsnorm = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"rmsnorm"] error:&error];
    psoSwiglu = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"swiglu"] error:&error];
    psoRope = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"rope"] error:&error];
    psoMultiheadAttention = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"multihead_attention"] error:&error];
    psoSoftmax = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"softmax"] error:&error];
    psoMultiheadWeightedSum = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"multihead_weighted_sum"] error:&error];

    buffers = [NSMutableDictionary dictionary];

    // create the first command buffer
    commandBuffer = [commandQueue commandBuffer];
    encoder = [commandBuffer computeCommandEncoder];
    return 0;
}

// Build a metal buffer object around the data
int lm_alloc(float **buf, unsigned long long size) {
    id<MTLBuffer> buffer = [device newBufferWithLength:size options:0];
    if (buffer == nil) {
        printf("Failed to allocate buffer of size %llu\n", size);
        exit(EXIT_FAILURE);
        return -1;
    }
    *buf = (float *)buffer.contents;
    buffers[@((uintptr_t)(*buf))] = buffer;
    return 0;
}

void lm_free(float *p) {
    [buffers removeObjectForKey:@((uintptr_t)p)];       // ARC will free the buffer
}

static id<MTLBuffer> mtl(float *p) {
    id<MTLBuffer> r = buffers[@((uintptr_t)p)];
    if (r == nil) {
        printf("Getting metal buffer for null pointer\n");
        exit(EXIT_FAILURE);
    }
    return r;
}

void lm_add(float *y, float *x, float *w, int n, int yoff, int xoff, int woff) {
    [encoder setComputePipelineState:psoAdd];
    [encoder setBuffer:mtl(y) offset:yoff*sizeof(float) atIndex:0];         // y
    [encoder setBuffer:mtl(x) offset:xoff*sizeof(float) atIndex:1];         // x
    [encoder setBuffer:mtl(w) offset:woff*sizeof(float) atIndex:2];      
    [encoder setBytes:&n length:sizeof(int) atIndex:4];                     // n
    MTLSize gridSize = MTLSizeMake(MIN(n, psoAdd.maxTotalThreadsPerThreadgroup), 1, 1);
    MTLSize threadgroupSize = gridSize;
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}

void lm_gemv(float *y, float *x, float *w, int n, int d, int yoff, int woff) {
    [encoder setComputePipelineState:psoGemv];
    [encoder setBuffer:mtl(y) offset:yoff*sizeof(float) atIndex:0];         // y
    [encoder setBuffer:mtl(x) offset:0 atIndex:1];                          // x
    [encoder setBuffer:mtl(w) offset:woff*sizeof(float) atIndex:2];         // w
    [encoder setBytes:&n length:sizeof(int) atIndex:3];                     // n
    [encoder setBytes:&d length:sizeof(int) atIndex:4];                     // d
    // gridSize = MTLSizeMake(oe->d, 1, 1);
    // threadgroupSize = MTLSizeMake(MIN(oe->d, psoGemv.maxTotalThreadsPerThreadgroup), 1, 1);
    MTLSize gridSize = MTLSizeMake(d*32, 1, 1);     // N_SIMDWIDTH = 32
    MTLSize threadgroupSize = MTLSizeMake(1024, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}

void lm_rmsnorm(float *y, float *x, float *w, int n, float eps, int woff) {
    [encoder setComputePipelineState:psoRmsnorm];
    [encoder setBuffer:mtl(y) offset:0 atIndex:0];                          // y
    [encoder setBuffer:mtl(x) offset:0 atIndex:1];                          // x
    [encoder setBuffer:mtl(w) offset:woff*sizeof(float) atIndex:2];         // w
    [encoder setBytes:&n length:sizeof(int) atIndex:3];                     // n
    [encoder setBytes:&eps length:sizeof(float) atIndex:4];                 // eps                
    [encoder setThreadgroupMemoryLength:32*sizeof(float) atIndex:0];        // buf
    MTLSize gridSize = MTLSizeMake(MIN(n, psoRmsnorm.maxTotalThreadsPerThreadgroup), 1, 1);
    MTLSize threadgroupSize = gridSize;
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}

void lm_swiglu(float *y, float *x, float *x2, int n) {
    [encoder setComputePipelineState:psoSwiglu];
    [encoder setBuffer:mtl(y) offset:0 atIndex:0];                          // y
    [encoder setBuffer:mtl(x) offset:0 atIndex:1];                          // x
    [encoder setBuffer:mtl(x2) offset:0 atIndex:2];                         // x2
    [encoder setBytes:&n length:sizeof(int) atIndex:3];                     // n
    MTLSize gridSize = MTLSizeMake(MIN(n/4, psoSwiglu.maxTotalThreadsPerThreadgroup), 1, 1);
    MTLSize threadgroupSize = gridSize;
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}

void lm_rope(float *q, float *k, int pos, int nq, int nk, int head_size, float theta, int koff) {
    [encoder setComputePipelineState:psoRope];
    [encoder setBuffer:mtl(q) offset:0 atIndex:0];                          // q
    [encoder setBuffer:mtl(k) offset:koff*sizeof(float) atIndex:1];         // k
    [encoder setBytes:&pos length:sizeof(int) atIndex:2];                   // pos
    [encoder setBytes:&nq length:sizeof(int) atIndex:3];                    // nq
    [encoder setBytes:&nk length:sizeof(int) atIndex:4];                    // nk
    [encoder setBytes:&head_size length:sizeof(int) atIndex:5];             // head_size
    [encoder setBytes:&theta length:sizeof(float) atIndex:6];               // theta
    MTLSize gridSize = MTLSizeMake(MIN(nq/2, psoRope.maxTotalThreadsPerThreadgroup), 1, 1);
    MTLSize threadgroupSize = gridSize;
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}

void lm_multihead_attention(float *att, float *q, float *k, int head_size, int n_heads, int d, int koff) {
    [encoder setComputePipelineState:psoMultiheadAttention];
    [encoder setBuffer:mtl(att) offset:0 atIndex:0];                        // att
    [encoder setBuffer:mtl(q) offset:0 atIndex:1];                          // q
    [encoder setBuffer:mtl(k) offset:koff*sizeof(float) atIndex:2];         // k
    [encoder setBytes:&head_size length:sizeof(int) atIndex:3];             // head_size
    [encoder setBytes:&n_heads length:sizeof(int) atIndex:4];               // n_heads
    [encoder setBytes:&d length:sizeof(int) atIndex:5];                     // d
    MTLSize gridSize = MTLSizeMake(d * n_heads, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(MIN(d * n_heads, psoMultiheadAttention.maxTotalThreadsPerThreadgroup), 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}

void lm_softmax(float *y, float *x, int n, int d) {
    [encoder setComputePipelineState:psoSoftmax];
    [encoder setBuffer:mtl(y) offset:0 atIndex:0];                          // y
    [encoder setBuffer:mtl(x) offset:0 atIndex:1];                          // x
    [encoder setBytes:&n length:sizeof(int) atIndex:2];                     // n
    MTLSize gridSize = MTLSizeMake(d, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(MIN(d, psoSoftmax.maxTotalThreadsPerThreadgroup), 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}

void lm_multihead_weighted_sum(float *y, float *att, float *v, int head_size, int n_heads, int d, int voff) {
    [encoder setComputePipelineState:psoMultiheadWeightedSum];
    [encoder setBuffer:mtl(y) offset:0 atIndex:0];                          // y
    [encoder setBuffer:mtl(att) offset:0 atIndex:1];                        // att
    [encoder setBuffer:mtl(v) offset:voff*sizeof(float) atIndex:2];         // v
    [encoder setBytes:&head_size length:sizeof(int) atIndex:3];             // head_size
    [encoder setBytes:&n_heads length:sizeof(int) atIndex:4];               // n_heads
    [encoder setBytes:&d length:sizeof(int) atIndex:5];                     // d
    MTLSize gridSize = MTLSizeMake(head_size * n_heads, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(MIN(head_size * n_heads, psoMultiheadWeightedSum.maxTotalThreadsPerThreadgroup), 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}

void lm_execute() {
    // finish encoding and commit command buffer
    [encoder endEncoding];
    [commandBuffer commit];                 
    [commandBuffer waitUntilCompleted];

    // create new command buffer
    commandBuffer = [commandQueue commandBuffer];
    encoder = [commandBuffer computeCommandEncoder];
}