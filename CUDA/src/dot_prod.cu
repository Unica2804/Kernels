#include <cuda_runtime.h>
#include <iostream>

#define MASK 0xFFFFFFFF

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__inline__ __device__ float warpReduceSum(float val){
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /=2){
        val += __shfl_down_sync(MASK, val, offset);

    }
    return val;
}

__global__ void dotProduct(const float * __restrict__ a, const float * __restrict__ b, float *result, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        atomicAdd(result, a[idx] * b[idx]);
    }
}

__global__ void dotProductoptimized(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ result, int N){
    extern __shared__ float warpsum[];
    size_t tid = threadIdx.x;
    size_t gid = blockIdx.x * blockDim.x + tid;
    int lane = tid % warpSize;
    int wid = tid / warpSize;
    int numWarps = blockDim.x / warpSize;
    float local_sum = 0.0f;

    if (gid < N){
        #pragma unroll
        for (size_t i = gid; i<N; i += blockDim.x * gridDim.x){
            local_sum += a[i] * b[i];
        }
    }

    //warp reduction
    local_sum = warpReduceSum(local_sum);

    //write warp result to shared memory in block level
    if(lane == 0){
        warpsum[wid] = local_sum;
    }
    __syncthreads();

    //block reduction by first warp
    if (wid == 0){
        float block_sum = (lane < numWarps) ? warpsum[lane] : 0.0f;
        block_sum = warpReduceSum(block_sum);
        if (lane == 0){
            atomicAdd(result, block_sum);
        }
    }
}

void applyDotProduct(const float *a, const float *b, float *result, int N){
    float *d_a, *d_b, *d_result;
    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_result, 0, sizeof(float)));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    int threadsPerBlock = blockSize;
    int warpsPerBlock = threadsPerBlock / 32;
    int sharedMemSize = warpsPerBlock * sizeof(float);
    // dotProduct<<<numBlocks, blockSize>>>(d_a, d_b, d_result, N);
    dotProductoptimized<<<numBlocks, blockSize, sharedMemSize>>>(d_a, d_b, d_result, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_result));
}

int main(){
    const size_t N = 10000000;
    float *a = new float[N];
    float *b = new float[N];
    for (size_t i = 0; i < N; ++i) {
        a[i] = 1.0f; 
        b[i] = 1.0f; 
    }
    float *result = new float[1];
    applyDotProduct(a, b, result, N);
    std::cout << "Dot product: " << result[0] << std::endl;
    delete[] a;
    delete[] b;
    delete[] result;
    return 0;
}