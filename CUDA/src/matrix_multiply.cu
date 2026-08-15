# include <iostream>
# include <cuda_runtime.h>
# include <curand_kernel.h>

__global__ void matmul(const float *A, const float *B, float *C, size_t N){
    size_t col = threadIdx.x + blockIdx.x * blockDim.x;
    size_t row = threadIdx.y + blockIdx.y * blockDim.y;
    float sum = 0.0f;
    if (row<N && col<N){
        for (size_t i=0; i<N; i++){
            sum += A[row*N+i] * B[i*N+col];
        }
        C[row*N+col]=sum;
    }
}

__global__ void init_matrix(float *M, size_t total, unsigned long long seed){
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < total) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, 0, &state);
        M[idx] = curand_uniform(&state);
    }
}

int main(){
    size_t N = 2048;
    size_t bytes = N*N*sizeof(float);
    float *h_A, *h_B, *h_C;
    cudaMallocManaged(&h_A, bytes);
    cudaMallocManaged(&h_B, bytes);
    cudaMallocManaged(&h_C, bytes);

    size_t total = N * N;
    constexpr int kThreads = 256;
    int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    init_matrix<<<blocks, kThreads>>>(h_A, total, 1234ULL);
    init_matrix<<<blocks, kThreads>>>(h_B, total, 5678ULL);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    int dev = 0;
    cudaGetDevice(&dev);
    cudaMemPrefetchAsync(h_A, bytes, dev);
    cudaMemPrefetchAsync(h_B, bytes, dev);
    cudaMemPrefetchAsync(h_C, bytes, dev);
    cudaDeviceSynchronize();

    matmul<<<gridDim, blockDim>>>(h_A, h_B, h_C, N);
    cudaDeviceSynchronize();

    cudaMemPrefetchAsync(h_C, bytes, cudaCpuDeviceId);
    cudaDeviceSynchronize();
    for (size_t i=0; i<10; i++){
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;


    cudaFree(h_A);
    cudaFree(h_B);
    cudaFree(h_C);
    return 0;
}