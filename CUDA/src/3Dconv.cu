#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
constexpr int MAX_KERNEL_SIZE = 10*10*10;

__constant__ float d_kernel[MAX_KERNEL_SIZE];

__global__ void conv3D(const float * __restrict__ input, float * __restrict__ output,
int column, int row, int depth, int kernel_column, int kernel_row, int kernel_depth, int stride, int padding){
    // Allocate constant memory for the kernel
    // Calculate the output index for the current thread
    size_t out_column = threadIdx.x + blockIdx.x * blockDim.x;
    size_t out_row = threadIdx.y + blockIdx.y * blockDim.y;
    size_t out_depth = threadIdx.z + blockIdx.z * blockDim.z;
    
    // Calculate the output dimensions
    int out_dim_column = ((column - kernel_column + 2 * padding) / stride) + 1;
    int out_dim_row = ((row - kernel_row + 2 * padding) / stride) + 1;
    int out_dim_depth = ((depth - kernel_depth + 2 * padding) / stride) + 1;

    // Check if the output index is within bounds
    if (out_column < out_dim_column && out_row < out_dim_row && out_depth < out_dim_depth) {
        float sum = 0.0f;
        // loop over the kernel and perform convolution
        #pragma unroll
        for (int kc = 0; kc < kernel_column; ++kc){
            #pragma unroll
            for (int kr = 0; kr < kernel_row; ++kr){
                #pragma unroll
                for (int kd = 0; kd < kernel_depth; ++kd){
                    // calculate the individual input index for column, row and depth
                    int in_column = (out_column * stride) - padding + kc;
                    int in_row = (out_row * stride) - padding + kr;
                    int in_depth = (out_depth * stride) - padding + kd;

                    // check if the input index is within bounds
                    if (in_column >= 0 && in_column < column && in_row >= 0 && in_row < row
                    && in_depth >= 0 && in_depth < depth){
                        // calculate the 1D index for the input
                        size_t input_idx = in_depth * (row * column) + in_row * column + in_column;

                        // calculate the 1D index for the kernel
                        size_t kernel_idx = kd * (kernel_row * kernel_column) + kr * kernel_column + kc;

                        // perform the convolution operation
                        sum += input[input_idx] * d_kernel[kernel_idx];
                    }

                }

                    
            }
        }   
        // calculate the 1D index for the output
        size_t output_idx = out_depth * (out_dim_row * out_dim_column) + out_row * out_dim_column + out_column;
        output[output_idx] = sum;
    }   

}

void apply_convolution(const float *h_input, float *h_output, const float *h_kernel, 
    int column, int row, int depth, int kernel_column, int kernel_row, int kernel_depth, int stride, int padding){
    float *d_input, *d_output;

    size_t input_size = column * row * depth * sizeof(float);
    size_t output_size = ((column - kernel_column + 2 * padding) / stride + 1) * ((row - kernel_row + 2 * padding) / stride + 1) * ((depth - kernel_depth + 2 * padding) / stride + 1) * sizeof(float);


    // Calculate output dimensions using the PyTorch formula
    int out_W = ((column - kernel_column + 2 * padding) / stride) + 1;
    int out_H = ((row - kernel_row + 2 * padding) / stride) + 1;
    int out_D = ((depth - kernel_depth + 2 * padding) / stride) + 1;

    CHECK_CUDA(cudaMalloc(&d_input, input_size));
    CHECK_CUDA(cudaMalloc(&d_output, output_size));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(d_kernel, h_kernel, kernel_column * kernel_row * kernel_depth * sizeof(float)));

    dim3 threadsPerBlock(8, 8, 8);
    int grid_x = (out_W + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int grid_y = (out_H + threadsPerBlock.y - 1) / threadsPerBlock.y;
    int grid_z = (out_D + threadsPerBlock.z - 1) / threadsPerBlock.z;

    dim3 numBlocks(grid_x, grid_y, grid_z);

    conv3D<<<numBlocks, threadsPerBlock>>>(d_input, d_output, column, row, depth, kernel_column, kernel_row, kernel_depth, stride, padding);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));    
}
int main() {
    // Parameters
    const size_t input_width = 256;
    const size_t kernel_size = 6;
    const size_t output_width = input_width - kernel_size + 1;

    // Allocate host memory
    float* h_input = new float[input_width * input_width * input_width];
    float* h_output = new float[output_width * output_width * output_width];
    float* h_kernel = new float[kernel_size * kernel_size * kernel_size];

    // Initialize input with some values
    for (size_t i = 0; i < input_width * input_width * input_width; ++i) {
        h_input[i] = static_cast<float>(i % 10);
    }

    // Initialize kernel with simple values 
    for (size_t i = 0; i < kernel_size * kernel_size * kernel_size; ++i) {
        h_kernel[i] = static_cast<float>(i + 1);
    }

    // Run convolution
    apply_convolution(h_input, h_output, h_kernel, input_width, input_width, input_width, kernel_size, kernel_size, kernel_size, 1, 0);

    // Print a small part of the output for verification
    printf("Output (first 5x5 block):\n");
    for (size_t i = 0; i < 5 && i < output_width; ++i) {
        for (size_t j = 0; j < 5 && j < output_width; ++j) {
            for (size_t k = 0; k < 5 && k < output_width; ++k) {

                printf("%6.1f ", h_output[i * output_width * output_width + j * output_width + k]);
            }
            printf("\n");               
        }
        printf("\n");
    }

    // Clean up host memory
    delete[] h_input;
    delete[] h_output;
    delete[] h_kernel;

    return 0;
}

