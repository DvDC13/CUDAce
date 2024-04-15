/*
    Compute the prefix sum using the decoupled look-back scan algorithm.
*/

/*
    Run:
        ./main
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

#include <cuda/atomic>

#define GPU_ERROR_CHECK(call) { gpu_assert((call), __FILE__, __LINE__); }

void gpu_assert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

void inclusive_scan_cpu(float* array, int size)
{
    for (int i = 1; i < size; i++)
    {
        array[i] += array[i - 1];
    }
}

__global__ void decoupledLookBack_scan(float* array, int* counter, cuda::std::atomic<char>* blockStates, float* blocksValueAggregate, int size)
{
    __shared__ int blockIndex;
    
    int tid = threadIdx.x;

    if (tid == 0) blockIndex = atomicAdd(counter, 1);

    __syncthreads();

    // Local Scan
    int gid = blockIndex * blockDim.x + tid;

    if (gid >= size) return;

    int index_offset = gid - blockIndex * blockDim.x;

    for (int d = 1; d < blockDim.x; d <<= 1)
    {
        int i_offset = index_offset - d;
        float tmp;
        if (i_offset >= 0)
            tmp = array[gid - d];
        __syncthreads();
        if (i_offset >= 0)
            array[gid] += tmp;
        __syncthreads();
    }

    // Set State
    if (blockIndex == 0)
    {
        blockStates[blockIndex].store('P', cuda::std::memory_order_release);
    }
    else
    {
        blockStates[blockIndex].store('A', cuda::std::memory_order_release);
    }

    blocksValueAggregate[blockIndex] = array[blockIndex * blockDim.x + blockDim.x - 1];

    __syncthreads();

    // Look Back
    if (blockIndex > 0)
    {
        float sum = 0.0f;
        int prevBlockIndex = blockIndex - 1;
        char state = blockStates[prevBlockIndex].load(cuda::std::memory_order_acquire);
        
        while (state != 'P')
        {
            if (state == 'A')
            {
                sum += blocksValueAggregate[prevBlockIndex];
                prevBlockIndex--;
            }

            state = blockStates[prevBlockIndex].load(cuda::std::memory_order_acquire);
        }

        sum += blocksValueAggregate[prevBlockIndex];

        atomicAdd(&array[gid], sum);

        __syncthreads();

        blockStates[blockIndex].store('P', cuda::std::memory_order_release);
    }
}

int main()
{
    const int size = 1 << 4;

    float* h_array = new float[size];
    float* d_result = new float[size];
    
    for (int i = 0; i < size; i++)
    {
        h_array[i] = 1.0f;
    }

    GPU_ERROR_CHECK(cudaSetDevice(0));

    dim3 block(2);
    dim3 grid(size / block.x);

    float* d_array;
    GPU_ERROR_CHECK(cudaMalloc((void**)&d_array, size * sizeof(float)));
    GPU_ERROR_CHECK(cudaMemcpy(d_array, h_array, size * sizeof(float), cudaMemcpyHostToDevice));

    int* d_counter;
    GPU_ERROR_CHECK(cudaMalloc((void**)&d_counter, sizeof(int)));
    GPU_ERROR_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    float* d_blocksValueAggregate;
    GPU_ERROR_CHECK(cudaMalloc((void**)&d_blocksValueAggregate, sizeof(float) * grid.x));
    GPU_ERROR_CHECK(cudaMemset(d_blocksValueAggregate, 0, sizeof(float) * grid.x));

    cuda::std::atomic<char>* d_blockStates;
    GPU_ERROR_CHECK(cudaMalloc((void**)&d_blockStates, sizeof(cuda::std::atomic<char>) * grid.x));
    GPU_ERROR_CHECK(cudaMemset(d_blockStates, 'X', sizeof(char) * grid.x));

    decoupledLookBack_scan<<<grid, block, sizeof(int)>>>(d_array, d_counter, d_blockStates, d_blocksValueAggregate, size);

    GPU_ERROR_CHECK(cudaMemcpy(d_result, d_array, size * sizeof(float), cudaMemcpyDeviceToHost));

    inclusive_scan_cpu(h_array, size);

    printf("CPU scan:\n");
    for (int i = 0; i < size; i++)
    {
        printf("%f ", h_array[i]);
    }
    printf("\n");


    printf("GPU scan:\n");
    for (int i = 0; i < size; i++)
    {
        printf("%f ", d_result[i]);
    }
    printf("\n");

    // Check the results
    for (int i = 0; i < size; i++)
    {
        if (h_array[i] != d_result[i])
        {
            fprintf(stderr, "Error: mismatch at index %d: %f != %f\n", i, h_array[i], d_result[i]);
            exit(EXIT_FAILURE);
        }
    }

    printf("Results match\n");

    delete[] h_array;
    delete[] d_result;

    GPU_ERROR_CHECK(cudaFree(d_array));

    return EXIT_SUCCESS;
}