# include <iostream>
# include <cuda_runtime.h>

__global__ void clip(const float *input, float *output, const float min_value, const float max_value, size_t N){
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
        float val = input[idx];
        output[idx] = val < min_value ? min_value : (val > max_value ? max_value : val);
    }
}

void apply_clipping(const float *input, float *output, const float min_value, const float max_value, size_t N){
    float *d_input, *d_output;
    size_t bytes = N * sizeof(float);
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    clip<<<numBlocks, blockSize>>>(d_input, d_output, min_value, max_value, N);
    // cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

int main(){
    const size_t N = 1000000;
    float *input = new float[N];
    float *output = new float[N];
    for(size_t i = 0; i < N; ++i){
        input[i] = static_cast<float>(i) / N * 200.0f - 100.0f; // Values between -100 and 100
    }
    float min_value = -10.0f;
    float max_value = 10.0f;
    apply_clipping(input, output, min_value, max_value, N);
    // Print some results for verification
    for(size_t i = 0; i < 10; ++i){
        std::cout << "input[" << i << "] = " << input[i] << ", output[" << i << "] = " << output[i] << '\n';
    }
    delete[] input;
    delete[] output;
    return 0;
}