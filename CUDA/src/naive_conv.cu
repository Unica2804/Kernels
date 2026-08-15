# include <iostream>
# include <cuda_runtime.h>

__global__ void naive_conv(const float *input, const float *kernel, float *output, size_t input_width, size_t kernel_width){
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < input_width - kernel_width + 1){
        float sum = 0.0f;
        for(size_t j = 0; j < kernel_width; ++j){
            sum += input[i + j] * kernel[j];
        }
        output[i] = sum;
    }
    
}

void convolve(const float *input, const float *kernel, float *output, size_t input_width, size_t kernel_width){
    float *d_input, *d_kernel, *d_output;
    size_t output_width = input_width - kernel_width +1;
    size_t input_bytes = input_width * sizeof(float);
    size_t kernel_bytes = kernel_width * sizeof(float);
    size_t output_bytes = output_width * sizeof(float);

    cudaMalloc((void**)&d_input, input_bytes);
    cudaMalloc((void**)&d_kernel, kernel_bytes);
    cudaMalloc((void**)&d_output, output_bytes);

    cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_bytes, cudaMemcpyHostToDevice);
    dim3 blockDim(256);
    dim3 gridDim((output_width + blockDim.x - 1) / blockDim.x);
    naive_conv<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, input_width, kernel_width);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

int main(){
    const size_t input_width = 1024;
    const size_t kernel_width = 5;
    const size_t output_width = input_width - kernel_width + 1;

    float *input = new float[input_width];
    float *kernel = new float[kernel_width];
    float *output = new float[output_width];

    for(size_t i = 0; i < input_width; ++i){
        input[i] = static_cast<float>(i);
    }
    for(size_t j = 0; j < kernel_width; ++j){
        kernel[j] = 1.0f;
        kernel[j+1] = 0.0f;
        kernel[j+2] = 2.0f;
        kernel[j+3] = 0.0f;
        kernel[j+4] = 1.0f;
    }

    convolve(input, kernel, output, input_width, kernel_width);

    std::cout << "Input: " << "\n";
    for(size_t i = 0; i < 10; ++i){
        std::cout << input[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Output of convolution: " << "\n";
    for(size_t i = 0; i < 10; ++i){
        std::cout << output[i] << " ";
    }
    std::cout << "\n";

    delete[] input;
    delete[] kernel;
    delete[] output;

    return 0;
}