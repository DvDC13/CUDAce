#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/device_vector.h"

#define XDIMENSION 1280
#define YDIMENSION 720

struct cuComplex
{
    float r;
    float i;

    __device__ cuComplex(float a, float b) : r(a), i(b) {}

    __device__ float lengthSquared() { return r * r + i * i; }

    __device__ cuComplex operator*(const cuComplex& a)
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    __device__ cuComplex operator+(const cuComplex& a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ int julia(int x, int y)
{
    const float scale = 1.5;
    float jx = scale * (float)(XDIMENSION / 2 - x) / (XDIMENSION / 2);
    float jy = scale * (float)(YDIMENSION / 2 - y) / (YDIMENSION / 2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i = 0; i < 100; i++)
    {
        a = a * a + c;
        if (a.lengthSquared() > 1000) return 0;
    }

    return 1;
}

__global__ void juliaKernel(unsigned char* ptr)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    int juliaValue = julia(x, y);
    ptr[offset * 3 + 0] = 255 * juliaValue;
    ptr[offset * 3 + 1] = 0;
    ptr[offset * 3 + 2] = 0;
}

int main()
{
    // Display julia in ppm format

    unsigned char* ptr = new unsigned char[XDIMENSION * YDIMENSION * 3];

    thrust::device_vector<unsigned char> dev_ptr(XDIMENSION * YDIMENSION * 3);

    auto start = std::chrono::high_resolution_clock::now();

    dim3 grid(XDIMENSION, YDIMENSION);
    juliaKernel<<<grid, 1>>>(thrust::raw_pointer_cast(dev_ptr.data()));

    thrust::copy(dev_ptr.begin(), dev_ptr.end(), ptr);

    auto end = std::chrono::high_resolution_clock::now();

    printf("Time taken: %lu ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    FILE* fp = fopen("julia.ppm", "wb");
    fprintf(fp, "P6\n%d %d\n255\n", XDIMENSION, YDIMENSION);
    fwrite(ptr, 1, XDIMENSION * YDIMENSION * 3, fp);

    fclose(fp);

    delete[] ptr;

    return EXIT_SUCCESS;
}