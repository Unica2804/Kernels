# include <iostream>
# include <cuda_runtime.h>

__global__ void geglu(const float *input, float *output, size_t N){
    size_t idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < N){
        float linear = input[idx];
        float gate = input[idx + N];
        output[idx] = 0.5f * linear * (1.0f + erff(gate * 0.70710678118f)); // 1/sqrt(2)
    }
}

void apply_geglu(const float *input, float *output, size_t N){
    float *d_input, *d_output;
    size_t input_bytes = 2 * N * sizeof(float);
    size_t output_bytes = N * sizeof(float);
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_output, output_bytes);
    cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    geglu<<<numBlocks, blockSize>>>(d_input, d_output, N);
    // cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

int main(){
    size_t N = 1000000;
    float *input = new float[2 * N];
    float *output = new float[N];
    for (size_t i = 0; i < 2 * N; ++i){
        input[i] = static_cast<float>(i) / (2 * N); // Example input
    }
    apply_geglu(input, output, N);
    // Print some results for verification
    for(size_t i = 0; i < 10; ++i){
        std::cout << "input[" << i << "] = " << input[i] << ", output[" << i << "] = " << output[i] << '\n';
    }
    delete[] input;
    delete[] output;
    return 0;
}