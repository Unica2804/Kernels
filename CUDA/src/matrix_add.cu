# include <iostream>
# include <cuda_runtime.h>

__global__ void matadd(const float *A, const float *B, float *C, size_t N, size_t M){
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < M*N){
        C[idx] = A[idx] + B[idx];
    }
}

void print_matrix(const float *M, size_t N, size_t M_cols){
    for (size_t i=0; i<N; i++){
        for (size_t j=0; j<M_cols; j++){
            std::cout << M[i*M_cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void add_matrices(const float *A, const float *B, float *C, size_t N, size_t M){
    size_t total_bytes = N * M * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, total_bytes);
    cudaMalloc((void**)&d_B, total_bytes);
    cudaMalloc((void**)&d_C, total_bytes);
    cudaMemcpy(d_A, A, total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, total_bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(256);
    dim3 gridDim((N*M + blockDim.x - 1) / blockDim.x);
    matadd<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, M);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, total_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(){
    size_t N = 2048;
    size_t M = 2048;
    size_t total = N * M;
    float *A = new float[total];
    float *B = new float[total];
    float *C = new float[total];
    unsigned long long seed = 12345ULL;
    for (size_t i=0; i<total; i++){
        A[i] = static_cast<float>((i+seed) % 100) / 100.0f;
        B[i] = static_cast<float>((i+seed+1) % 100) / 100.0f;
    }
    add_matrices(A, B, C, N, M);
    std::cout << "First 10 elements of the result matrix C:" << "\n";
    for (int i = 0; i < 10; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << "\n";
    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}