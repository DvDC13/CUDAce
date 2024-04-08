/*
    Calculate the sum of the Taylor series of the sine function for a given number of steps and terms between 0 and PI.
    The Taylor series is calculated using the formula:
        sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...
    The result is calculated using the trapezoidal rule for integration correction.
*/

/*
    Run:
        ./main 1000000 1000 4
*/

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/device_vector.h"

using namespace std::chrono;

const double PI = 3.14159265358979323846f;

__host__ __device__ inline float sinusTaylor(float x, int terms) {
    
    // sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...

    float term = x;
    float sum = term;
    float x2 = x*x;
    for (int n = 1; n < terms; n++) {
        term *= -x2 / (float) (2 * n * (2 * n + 1));
        sum += term;
    }
    return sum;
}

void singleThreadedCPU(int steps, int terms) {

    double step_size = PI / (steps - 1);

    auto start = high_resolution_clock::now();

    double cpu_sum = 0.0;
    for (int i = 0; i < steps; i++) {
        float x = i * step_size;
        cpu_sum += sinusTaylor(x, terms);
    }

    // Using trapzoidal rule for integration correction
    cpu_sum -= (sinusTaylor(0.0f, terms) + sinusTaylor(PI, terms)) / 2.0;
    cpu_sum *= step_size;

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(stop - start);

    printf("Result: %.10f\n", cpu_sum);
    printf("Threads: 1\n");
    printf("Time taken by function: %lu milliseconds\n", duration.count());
}

void multiThreadedCPU(int steps, int terms, int threads) {

    double step_size = PI / (steps - 1);

    auto start = high_resolution_clock::now();

    double cpu_sum = 0.0;

    omp_set_num_threads(threads);
    #pragma omp parallel for reduction(+:cpu_sum)
    for (int i = 0; i < steps; i++) {
        float x = i * step_size;
        cpu_sum += sinusTaylor(x, terms);
    }

    // Using trapzoidal rule for integration correction
    cpu_sum -= (sinusTaylor(0.0f, terms) + sinusTaylor(PI, terms)) / 2.0;
    cpu_sum *= step_size;

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(stop - start);

    printf("Result: %.10f\n", cpu_sum);
    printf("Threads: %d\n", threads);
    printf("Time taken by function: %lu milliseconds\n", duration.count());
}

__global__ void sinusTaylorGPU(float *d_local_sum, int steps, int terms, double step_size) {

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= steps) return;

    float x = gid * step_size;
    d_local_sum[gid] = sinusTaylor(x, terms);
}

void multiThreadedGPU(int steps, int terms) {

    int threads = 1024;

    int blocks = (steps + threads - 1) / threads;

    double step_size = PI / (steps - 1);

    thrust::device_vector<float> d_local_sum(steps, 0.0f);

    auto start = high_resolution_clock::now();

    sinusTaylorGPU<<<blocks, threads>>>(thrust::raw_pointer_cast(d_local_sum.data()), steps, terms, step_size);

    double gpu_sum = thrust::reduce(d_local_sum.begin(), d_local_sum.end(), 0.0f, thrust::plus<float>());

    // Using trapzoidal rule for integration correction
    gpu_sum -= (sinusTaylor(0.0f, terms) + sinusTaylor(PI, terms)) / 2.0;
    gpu_sum *= step_size;

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(stop - start);

    printf("Result: %.10f\n", gpu_sum);
    printf("Threads: %d\n", threads);
    printf("Time taken by function: %lu milliseconds\n", duration.count());
}

// ./exe 1000000 1000 4
int main(int argc, char *argv[]) {

    int steps = atoi(argv[1]);
    int terms = atoi(argv[2]);

    int threads = (argc > 3) ? atoi(argv[3]) : 1;

    printf("Single-threaded CPU:\n");
    singleThreadedCPU(steps, terms);

    printf("\nMulti-threaded CPU:\n");
    multiThreadedCPU(steps, terms, threads);

    printf("\nMulti-threaded GPU:\n");
    multiThreadedGPU(steps, terms);

    return EXIT_SUCCESS;
}