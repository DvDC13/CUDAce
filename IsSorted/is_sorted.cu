/*
    Check if an array is sorted in ascending order
*/

/*
    ./main <size>
*/

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/device_vector.h"

__global__ void is_sorted(int* d_in, int* d_out, int size)
{
    extern __shared__ int s_in[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    int val = 0;
    if (gid < size - 1)
    {
        val = d_in[gid] > d_in[gid + 1] ? 1 : 0;
    }

    s_in[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_in[tid] += s_in[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(d_out, s_in[0]);
    }
}

int main(int argc, char *argv[])
{
    int size = argc > 1 ? atoi(argv[1]) : 1000000;

    int* h_in = new int[size];
    
    for (int i = 0; i < size; i++)
    {
        h_in[i] = i;
    }

    int* d_in;
    cudaMalloc(&d_in, size * sizeof(int));
    cudaMemcpy(d_in, h_in, size * sizeof(int), cudaMemcpyHostToDevice);

    int* d_out;
    cudaMalloc(&d_out, sizeof(int));
    cudaMemset(d_out, 0, sizeof(int));

    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);

    auto start = std::chrono::high_resolution_clock::now();
    is_sorted<<<grid, block, sizeof(int) * block.x>>>(d_in, d_out, size);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    printf("GPU time: %f ms\n", duration.count());

    int h_out;
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Array is sorted: %s\n", h_out == 0 ? "true" : "false");

    cudaFree(d_in);

    delete[] h_in;

    cudaDeviceReset();

    return EXIT_SUCCESS;
}