/*
    Add 2 vectors
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

void add_vec(int* a, int* b, int* c, int size)
{
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_vec_cuda(int* a, int* b, int* c, int size)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < size)
    {
        c[gid] = a[gid] + b[gid];
        gid += blockDim.x * gridDim.x;
    }
}

int main(int argc, char* argv[])
{
    int size = argc > 1 ? atoi(argv[1]) : 1000000;

    int* a = (int*)malloc(size * sizeof(int));
    int* b = (int*)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    int* c_cpu = (int*)malloc(size * sizeof(int));
    int* c_gpu = (int*)malloc(size * sizeof(int));

    auto start = std::chrono::high_resolution_clock::now();
    add_vec(a, b, c_cpu, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    printf("CPU time: %f ms\n", duration.count());

    int* d_a;
    int* d_b;
    int* d_c;

    cudaMalloc(&d_a, size * sizeof(int));
    cudaMalloc(&d_b, size * sizeof(int));
    cudaMalloc(&d_c, size * sizeof(int));

    cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 1024;
    int blocks = (size + threads - 1) / threads;

    start = std::chrono::high_resolution_clock::now();
    add_vec_cuda<<<blocks, threads>>>(d_a, d_b, d_c, size);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    printf("GPU time: %f ms\n", duration.count());

    cudaMemcpy(c_gpu, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++)
    {
        if (c_cpu[i] != c_gpu[i])
        {
            printf("Error at %d\n", i);
            printf("CPU: %d\n", c_cpu[i]);
            printf("GPU: %d\n", c_gpu[i]);
            break;
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(a);
    free(b);
    free(c_cpu);
    free(c_gpu);

    return EXIT_SUCCESS;
}