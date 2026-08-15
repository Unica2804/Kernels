# include <cuda_runtime.h>
# include <iostream>

__global__ void siLU(const float *input, float *output, size_t N){
    size_t idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < N){
        float x = input[idx];
        output[idx]=x/(1+expf(-x));
    }
}

void apply_silu(const float *input, float *output, size_t N){
    float *d_input, *d_output;
    size_t bytes = N*sizeof(float);
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);
    dim3 blockDim(256);
    dim3 gridDim((N+blockDim.x-1)/blockDim.x);
    siLU<<<gridDim, blockDim>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

int main(){
    const size_t N = 1000000;
    float *input = new float[N];
    float *output = new float[N];
    for(size_t i = 0; i < N; ++i){
        input[i] = static_cast<float>(i) - 100.0f; // Range from -500k to +500k
    }
    apply_silu(input, output, N);
    // Print first 10 results for verification
    for(size_t i = 0; i < 20; ++i){
        std::cout << "SiLU(" << input[i] << ") = " << output[i] << '\n';
    }
    delete[] input;
    delete[] output;
    return 0;
}