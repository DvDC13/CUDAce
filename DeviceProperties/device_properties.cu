/*
    Query the device on your machine
*/

/*
    ./main
*/

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main()
{
    cudaDeviceProp prop;

    int count;
    cudaGetDeviceCount(&count);

    for (int i = 0; i < count; i++)
    {
        cudaGetDeviceProperties(&prop, i);

        printf("****General Information****\n");
        printf("Device Number: %d\n", i);
        printf("Device Name: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Clock Rate: %d\n", prop.clockRate);
        printf("Device Copy Overlap: %s\n", prop.deviceOverlap ? "Enabled" : "Disabled");
        printf("Kernel Execution Timeout: %s\n\n", prop.kernelExecTimeoutEnabled ? "Enabled" : "Disabled");

        printf("****Memory Information****\n");
        printf("Total Global Memory: %lu\n", prop.totalGlobalMem);
        printf("Total Constant Memory: %lu\n", prop.totalConstMem);
        printf("Max Memory Pitch: %lu\n", prop.memPitch);
        printf("Texture Alignment: %lu\n\n", prop.textureAlignment);

        printf("****Multiprocessor Information****\n");
        printf("Multiprocessor Count: %d\n", prop.multiProcessorCount);
        printf("Shared Memory Per Multiprocessor: %lu\n", prop.sharedMemPerBlock);
        printf("Registers Per Multiprocessor: %d\n", prop.regsPerBlock);
        printf("Threads Per Warp: %d\n", prop.warpSize);
        printf("Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
        printf("Max Threads Dimension: %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max Grid Size: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }


    return EXIT_SUCCESS;
}