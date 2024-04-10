/*
    Choose the device based on the needs
*/

/*
    ./main
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

int main()
{
    cudaDeviceProp prop;
    int dev;

    cudaGetDevice(&dev);
    
    printf("ID of current CUDA device: %d\n", dev);

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 7;
    prop.minor = 5;

    cudaChooseDevice(&dev, &prop);
    printf("ID of CUDA device closest to revision 7.5: %d\n", dev);

    cudaSetDevice(dev);

    return EXIT_SUCCESS;
}