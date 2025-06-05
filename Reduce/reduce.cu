/*
    Compute the sum of an array
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

// Interleaved Addressing
__global__ void reduce0(int* d_in, int* d_out, int size)
{
    extern __shared__ int s_in[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    s_in[tid] = d_in[gid];
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
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

// Interleaved Addressing without branch divergence and % operation
__global__ void reduce1(int* d_in, int* d_out, int size)
{
    extern __shared__ int s_in[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    s_in[tid] = d_in[gid];
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;
        if (index < blockDim.x)
        {
            s_in[index] += s_in[index + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(d_out, s_in[0]);
    }
}

void sum_cpu(int* h_in, int& h_out, int size)
{
    for (int i = 0; i < size; i++)
    {
        h_out += h_in[i];
    }
}

int main(int argc, char** argv)
{
    int size = argc > 1 ? atoi(argv[1]) : 1000000;

    int* h_in = new int[size];
    
    for (int i = 0; i < size; i++)
    {
        h_in[i] = i;
    }

    int h_out_cpu = 0;

    auto start = std::chrono::high_resolution_clock::now();
    sum_cpu(h_in, h_out_cpu, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    printf("CPU time: %f ms\n", duration.count());

    int* d_in;
    cudaMalloc(&d_in, size * sizeof(int));
    cudaMemcpy(d_in, h_in, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);

    int* d_out_reduce0;
    cudaMalloc(&d_out_reduce0, sizeof(int));
    
    // Interleaved Addressing
    start = std::chrono::high_resolution_clock::now();
    reduce0<<<grid, block, sizeof(int) * block.x>>>(d_in, d_out_reduce0, size);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    printf("GPU reduce0 time: %f ms\n", duration.count());

    int* d_out_reduce1;
    cudaMalloc(&d_out_reduce1, sizeof(int));

    // Interleaved Addressing without branch divergence and % operation
    start = std::chrono::high_resolution_clock::now();
    reduce1<<<grid, block, sizeof(int) * block.x>>>(d_in, d_out_reduce1, size);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    printf("GPU reduce1 time: %f ms\n", duration.count());

    int* d_out_reduce0_gpu = new int;
    cudaMemcpy(d_out_reduce0_gpu, d_out_reduce0, sizeof(int), cudaMemcpyDeviceToHost);

    int* d_out_reduce1_gpu = new int;
    cudaMemcpy(d_out_reduce1_gpu, d_out_reduce1, sizeof(int), cudaMemcpyDeviceToHost);

    printf("CPU: %d\n", h_out_cpu);
    printf("GPU reduce0: %d\n", *d_out_reduce0_gpu);
    printf("GPU reduce1: %d\n", *d_out_reduce1_gpu);

    delete[] h_in;

    cudaFree(d_in);
    cudaFree(d_out_reduce0);
    cudaFree(d_out_reduce1);

    return EXIT_SUCCESS; 
}