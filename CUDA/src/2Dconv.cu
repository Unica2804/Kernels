#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#include <iostream>
#include <cuda_runtime.h>

#define KERNEL_SIZE 3      
#define BLOCK_DIM 16
#define OVERLAP (BLOCK_DIM - KERNEL_SIZE - 1) 

__constant__ float d_kernel[KERNEL_SIZE * KERNEL_SIZE];

__global__ void conv2d(const float * __restrict__ input, float * __restrict__ output, int input_width){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < input_width && col < input_width){
        float sum = 0.0f;

        #pragma unroll
        for (int i = 0; i< KERNEL_SIZE; ++i){
            for (int j = 0; j < KERNEL_SIZE; ++j){
                int inputidx = (row + i) * input_width + (col + j);
                sum += input[inputidx] * d_kernel[i * KERNEL_SIZE + j];
            }
        }

        int output_idx = row * input_width + col;
        output[output_idx] = sum;
    }
}

__global__ void conv2d_shared(const float * __restrict__ input, float * __restrict__ output, 
    int input_width, int output_width){
        // Usint a tile of shared memory to hold the input patch for convolution
        __shared__ float s_tile[BLOCK_DIM][BLOCK_DIM];

        size_t op_x = blockIdx.x * OVERLAP + threadIdx.x;
        size_t op_y = blockIdx.y * OVERLAP + threadIdx.y;

        size_t ip_x = blockIdx.x * OVERLAP + threadIdx.x;
        size_t ip_y = blockIdx.y * OVERLAP + threadIdx.y;

        if (ip_x < input_width && ip_y < input_width){
            s_tile[threadIdx.y][threadIdx.x] = input[ip_y * input_width + ip_x];
        }
        else {
            s_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        if (op_x < output_width && op_y < output_width){
            float sum = 0.0f;

            #pragma unroll
            for (int f_row = 0; f_row < KERNEL_SIZE; ++f_row){
                #pragma unroll
                for (int f_col = 0; f_col < KERNEL_SIZE; ++f_col){
                    size_t tile_row = threadIdx.y + f_row;
                    size_t tile_col = threadIdx.x + f_col;
                    if(tile_row < BLOCK_DIM && tile_col < BLOCK_DIM){
                        sum += s_tile[tile_row][tile_col] * d_kernel[f_row * KERNEL_SIZE + f_col];
                    }
                }
            }
            output[op_y * output_width + op_x] = sum;
        }

    }

void apply_convolution(const float *h_input, float *h_output, const float *h_kernel, int input_width, int output_width){
    float *d_input, *d_output;

    size_t input_size = input_width * input_width * sizeof(float);
    size_t output_size = output_width * output_width * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_input, input_size));
    CHECK_CUDA(cudaMalloc(&d_output, output_size));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(d_kernel, h_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float)));

    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((output_width + blockDim.x - 1) / blockDim.x, (output_width + blockDim.y - 1) / blockDim.y);

    conv2d_shared<<<gridDim, blockDim, BLOCK_DIM * BLOCK_DIM * sizeof(float)>>>(d_input, d_output, input_width, output_width);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));    
}
int main() {
    // Parameters
    const int input_width = 4096;
    const int kernel_size = KERNEL_SIZE;
    const int output_width = input_width - kernel_size + 1;

    // Allocate host memory
    float* h_input = new float[input_width * input_width];
    float* h_output = new float[output_width * output_width];
    float* h_kernel = new float[kernel_size * kernel_size];

    // Initialize input with some values
    for (int i = 0; i < input_width * input_width; ++i) {
        h_input[i] = static_cast<float>(i % 10);
    }

    // Initialize kernel with simple values (e.g., 1,2,3,...)
    for (int i = 0; i < kernel_size * kernel_size; ++i) {
        h_kernel[i] = static_cast<float>(i + 1);
    }

    // Run convolution
    apply_convolution(h_input, h_output, h_kernel, input_width, output_width);

    // Print a small part of the output for verification
    printf("Output (first 5x5 block):\n");
    for (int i = 0; i < 5 && i < output_width; ++i) {
        for (int j = 0; j < 5 && j < output_width; ++j) {
            printf("%6.1f ", h_output[i * output_width + j]);
        }
        printf("\n");
    }

    // Clean up host memory
    delete[] h_input;
    delete[] h_output;
    delete[] h_kernel;

    return 0;
}