/*
    Print "Hello from CPU" from CPU and "Hello from GPU" from GPU.
*/

/*
    Run:
        ./main <number of threads per block> <number of blocks>
*/


#include <stdio.h>
#include <chrono>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void helloFromGPU()
{
    printf("Hello from GPU\n");
}

int main(int argc, char *argv[])
{
    int numberOfThreads = argc > 2 ? atoi(argv[2]) : 1;
    int numberOfBlocks = argc > 1 ? atoi(argv[1]) : 1;

    printf("Hello from CPU\n");

    if (numberOfThreads > 1024)
    {
        printf("Number of threads per block must be less than 1024\n");
        return EXIT_FAILURE;
    }

    helloFromGPU<<<numberOfBlocks, numberOfThreads>>>();

    cudaDeviceReset();

    return EXIT_SUCCESS;
}