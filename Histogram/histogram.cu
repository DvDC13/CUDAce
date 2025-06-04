/*
    Compute the histogram of an image
*/

/*
    Run:
        ./main <image filename>
*/

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../Helpers/stb_image.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/device_vector.h"

__global__ void histogram_gmem_atomic(unsigned char* d_in, int width, int height, unsigned int* d_out)
{    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int nt = blockDim.y * blockDim.x;

    int bid = blockIdx.y * gridDim.x + blockIdx.x;

    unsigned int *gmem = d_out + bid * 3 * 256;
    for (int i = tid; i < 3 * 256; i += nt)
    {
        gmem[i] = 0;
    }

    for (int col = x; col < width; col += nx)
    {
        for (int row = y; row < height; row += ny)
        {
            int idx = (row * width + col) * 3;
            unsigned int r = d_in[idx];
            unsigned int g = d_in[idx + 1];
            unsigned int b = d_in[idx + 2];

            atomicAdd(&gmem[256 * 0 + r], 1);
            atomicAdd(&gmem[256 * 1 + g], 1);
            atomicAdd(&gmem[256 * 2 + b], 1);
        }
    }
}

__global__ void histogram_final_accum(const unsigned int* d_in, int n, unsigned int* d_out)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= 3 * 256) return;

    unsigned int sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += d_in[gid + 3 * 256 * i];
    }

    d_out[gid] = sum;
}

void histogram_cpu(unsigned char* h_in, int width, int height, unsigned int* h_out)
{
    for (int i = 0; i < width * height; i++)
    {
        unsigned int r = (unsigned int)(h_in[i * 3]);
        unsigned int g = (unsigned int)(h_in[i * 3 + 1]);
        unsigned int b = (unsigned int)(h_in[i * 3 + 2]);
        h_out[256 * 0 + r]++;
        h_out[256 * 1 + g]++;
        h_out[256 * 2 + b]++;
    }
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <image filename>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int width, height, channels;
    unsigned char *data = stbi_load(argv[1], &width, &height, &channels, 3);

    if (!data)
    {
        std::cerr << "Failed to load image: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Image loaded: " << width << "x" << height << ", channels: " << channels << std::endl;

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    unsigned char* d_in;
    cudaMalloc(&d_in, width * height * 3);
    cudaMemcpy(d_in, data, width * height * 3, cudaMemcpyHostToDevice);

    unsigned int* d_out;
    cudaMalloc(&d_out, sizeof(unsigned int) * 3 * 256);
    cudaMemset(d_out, 0, sizeof(unsigned int) * 3 * 256);

    int num_blocks = grid.x * grid.y;

    unsigned int* d_out_blocks;
    cudaMalloc(&d_out_blocks, sizeof(unsigned int) * 3 * 256 * num_blocks);
    cudaMemset(d_out_blocks, 0, sizeof(unsigned int) * 3 * 256 * num_blocks);

    auto start = std::chrono::high_resolution_clock::now();

    histogram_gmem_atomic<<<grid, block>>>(d_in, width, height, d_out_blocks);
    cudaDeviceSynchronize();

    histogram_final_accum<<<num_blocks * 3, 256>>>(d_out_blocks, num_blocks, d_out);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start;

    unsigned int h_out_gpu[3 * 256];
    cudaMemcpy(h_out_gpu, d_out, sizeof(unsigned int) * 3 * 256, cudaMemcpyDeviceToHost);

    unsigned int h_out_cpu[3 * 256];
    memset(h_out_cpu, 0, sizeof(unsigned int) * 3 * 256);
    histogram_cpu(data, width, height, h_out_cpu);

    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;

    for (int i = 0; i < 3 * 256; i++)
    {
        if (h_out_cpu[i] != h_out_gpu[i])
        {
            std::cerr << "Error: " << i << " " << h_out_cpu[i] << " " << h_out_gpu[i] << std::endl;
        }
    }

    std::cout << "Done" << std::endl;

    stbi_image_free(data);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_out_blocks);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}