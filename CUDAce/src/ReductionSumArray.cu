#include "ReductionSumArray.cuh"

#include "Error.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cstdlib>

/* 
* This function is used to calculate the sum of an array on the GPU.
* Method: Interleaved Addressing.
* Problems:
* Highly divergent warps are not efficient.
* "%" operator is slow.
*/
__global__ void ReductionSumArrayKernel1(float* input_array, float* output_array)
{
	// Create a shared memory array
	extern __shared__ float shared_array[];
	
	// Get the index of the thread
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	// Load the input array into the shared memory
	shared_array[threadIdx.x] = input_array[index];
	
	// Synchronize the threads
	__syncthreads();

	// Calculate the sum of the array
	for (unsigned int i = 1; i < blockDim.x; i *= 2)
	{
		if (threadIdx.x % (2 * i) == 0)
			shared_array[threadIdx.x] += shared_array[threadIdx.x + i];
		__syncthreads();
	}

	// Store the result in the output array
	if (threadIdx.x == 0) output_array[blockIdx.x] = shared_array[0];
}

/*
* This function is used to calculate the sum of an array on the GPU.
* Method: Interleaved Addressing.
* Problem:
* Shared memory bank conflicts.
*/
__global__ void ReductionSumArrayKernel2(float* input_array, float* output_array)
{
	// Create a shared memory array
	extern __shared__ float shared_array[];

	// Get the index of the thread
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	// Load the input array into the shared memory
	shared_array[threadIdx.x] = input_array[index];

	// Synchronize the threads
	__syncthreads();

	// Calculate the sum of the array
	for (unsigned int i = 1; i < blockDim.x; i *= 2)
	{
		int index = 2 * i * threadIdx.x;
		if (index < blockDim.x) shared_array[index] += shared_array[index + i];
		__syncthreads();
	}

	// Store the result in the output array
	if (threadIdx.x == 0) output_array[blockIdx.x] = shared_array[0];
}

/*
* This function is used to calculate the sum of an array on the GPU.
* Method: Sequential Addressing.
* Problems:
* Sequential addressing is conflict free.
* Half of the threads are idle on the first iteration.
*/
__global__ void ReductionSumArrayKernel3(float* input_array, float* output_array)
{
	// Create a shared memory array
	extern __shared__ float shared_array[];

	// Get the index of the thread
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	// Load the input array into the shared memory
	shared_array[threadIdx.x] = input_array[index];

	// Synchronize the threads
	__syncthreads();

	// Calculate the sum of the array
	for (unsigned int i = blockDim.x / 2; i > 0; i /= 2)
	{
		if (threadIdx.x < i) shared_array[threadIdx.x] += shared_array[threadIdx.x + i];
		__syncthreads();
	}

	// Store the result in the output array
	if (threadIdx.x == 0) output_array[blockIdx.x] = shared_array[0];
}

/*
* This function is used to calculate the sum of an array on the GPU.
* Method: Sequential Addressing.
* Problem:
* Not unrolling the Last Warp
*/
__global__ void ReductionSumArrayKernel4(float* input_array, float* output_array)
{
	// Create a shared memory array
	extern __shared__ float shared_array[];

	// Get the index of the thread
	int index = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Load the input array into the shared memory
	shared_array[threadIdx.x] = input_array[index] + input_array[index + blockDim.x];

	// Synchronize the threads
	__syncthreads();

	// Calculate the sum of the array
	// Calculate the sum of the array
	for (unsigned int i = blockDim.x / 2; i > 0; i /= 2)
	{
		if (threadIdx.x < i) shared_array[threadIdx.x] += shared_array[threadIdx.x + i];
		__syncthreads();
	}

	// Store the result in the output array
	if (threadIdx.x == 0) output_array[blockIdx.x] = shared_array[0];
}

/*
* This function is used to calculate the sum of an array on the GPU.
* Method: Sequential Addressing.
* Problem:
* Not knowing the number of interation at compile time.
*/
__global__ void ReductionSumArrayKernel5(float* input_array, float* output_array)
{
	// Create a shared memory array
	extern __shared__ float shared_array[];

	// Get the index of the thread
	int index = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Load the input array into the shared memory
	shared_array[threadIdx.x] = input_array[index] + input_array[index + blockDim.x];

	// Synchronize the threads
	__syncthreads();

	// Calculate the sum of the array
	for (unsigned int i = blockDim.x / 2; i > 32; i /= 2)
	{
		if (threadIdx.x < i) shared_array[threadIdx.x] += shared_array[threadIdx.x + i];
		__syncthreads();
	}

	// Unrolling the last warp
	if (threadIdx.x < 32)
	{
		shared_array[threadIdx.x] += shared_array[threadIdx.x + 32];
		shared_array[threadIdx.x] += shared_array[threadIdx.x + 16];
		shared_array[threadIdx.x] += shared_array[threadIdx.x + 8];
		shared_array[threadIdx.x] += shared_array[threadIdx.x + 4];
		shared_array[threadIdx.x] += shared_array[threadIdx.x + 2];
		shared_array[threadIdx.x] += shared_array[threadIdx.x + 1];
	}

	// Store the result in the output array
	if (threadIdx.x == 0) output_array[blockIdx.x] = shared_array[0];
}

/*
* This function is used to calculate the sum of the last warp of an array on the GPU.
*/
template <int blockSize>
__device__ void WarpReduce(volatile float* sdata, unsigned int index)
{
	if (blockSize >= 64) sdata[index] += sdata[index + 32];
	if (blockSize >= 32) sdata[index] += sdata[index + 16];
	if (blockSize >= 16) sdata[index] += sdata[index + 8];
	if (blockSize >= 8) sdata[index] += sdata[index + 4];
	if (blockSize >= 4) sdata[index] += sdata[index + 2];
	if (blockSize >= 2) sdata[index] += sdata[index + 1];
}

/*
* This function is used to calculate the sum of an array on the GPU.
* Method: Sequential Addressing.
*/
template <int blockSize>
__global__ void ReductionSumArrayKernel6(float* input_array, float* output_array)
{
	// Create a shared memory array
	extern __shared__ float shared_array[];

	// Get the index of the thread
	int index = blockIdx.x * (blockSize * 2) + threadIdx.x;

	// Load the input array into the shared memory
	shared_array[threadIdx.x] = input_array[index] + input_array[index + blockSize];

	// Synchronize the threads
	__syncthreads();

	// Calculate the sum of the array
	if (blockSize >= 1024)
	{ 
		if (threadIdx.x < 512) { shared_array[threadIdx.x] += shared_array[threadIdx.x + 512]; }
		__syncthreads();
	}

	if (blockSize >= 512)
	{
		if (threadIdx.x < 256) shared_array[threadIdx.x] += shared_array[threadIdx.x + 256];
		__syncthreads();
	}

	if (blockSize >= 256)
	{
		if (threadIdx.x < 128) shared_array[threadIdx.x] += shared_array[threadIdx.x + 128];
		__syncthreads();
	}

	if (blockSize >= 128)
	{
		if (threadIdx.x < 64) shared_array[threadIdx.x] += shared_array[threadIdx.x + 64];
		__syncthreads();
	}

	if (threadIdx.x < 32) WarpReduce<blockSize>(shared_array, threadIdx.x);

	// Store the result in the output array
	if (threadIdx.x == 0) output_array[blockIdx.x] = shared_array[0];
}

/*
* This function is used to calculate the sum of an array on the GPU.
* Method: Sequential Addressing.
*/
template <int blockSize>
__global__ void ReductionSumArrayKernel7(float* input_array, float* output_array, int arraySize)
{
	// Create a shared memory array
	extern __shared__ float shared_array[];

	// Get the index of the thread
	int index = blockIdx.x * (blockSize * 2) + threadIdx.x;

	// Get the grid size
	int gridSize = blockSize * 2 * gridDim.x;

	// Load the input array into the shared memory
	shared_array[threadIdx.x] = 0;
	while (index < arraySize)
	{
		shared_array[threadIdx.x] += input_array[index] + input_array[index + blockSize];
		index += gridSize;
	}

	// Synchronize the threads
	__syncthreads();

	// Calculate the sum of the array
	if (blockSize >= 1024)
	{
		if (threadIdx.x < 512) { shared_array[threadIdx.x] += shared_array[threadIdx.x + 512]; }
		__syncthreads();
	}
	
	if (blockSize >= 512)
	{
		if (threadIdx.x < 256) shared_array[threadIdx.x] += shared_array[threadIdx.x + 256];
		__syncthreads();
	}

	if (blockSize >= 256)
	{
		if (threadIdx.x < 128) shared_array[threadIdx.x] += shared_array[threadIdx.x + 128];
		__syncthreads();
	}

	if (blockSize >= 128)
	{
		if (threadIdx.x < 64) shared_array[threadIdx.x] += shared_array[threadIdx.x + 64];
		__syncthreads();
	}

	if (threadIdx.x < 32) WarpReduce<blockSize>(shared_array, threadIdx.x);

	// Store the result in the output array
	if (threadIdx.x == 0) output_array[blockIdx.x] = shared_array[0];
}

/*
* This function is used to calculate the sum of an array.
* 7 methods are used to calculate the sum of the array.
* We use the last method to calculate the sum of the array.
* The last method is the fastest method.
*/
void ReductionSumArray(int arraySize)
{
	// Create the input and output arrays
	float* input_array = new float[arraySize];
	float* output_array = new float[arraySize];

	// Fill the input array with random numbers
	for (int i = 0; i < arraySize; i++) input_array[i] = (float)i;

	// Create the input and output arrays on the GPU
	float* d_input_array;
	float* d_output_array;
	cudaMalloc((void**)&d_input_array, arraySize * sizeof(float));
	cudaMalloc((void**)&d_output_array, arraySize * sizeof(float));

	// Copy the input array to the GPU
	cudaMemcpy(d_input_array, input_array, arraySize * sizeof(float), cudaMemcpyHostToDevice);

	// Define the block size
	const int blockSize = 1024;
	int gridSize = (arraySize + blockSize - 1) / blockSize;
	int sharedMemorySize = blockSize * sizeof(float);

	// Create the events to measure the time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	switch (blockSize)
	{
	case 1024:
		// Calculate the sum of the array
		cudaEventRecord(start);
		ReductionSumArrayKernel7<blockSize> << <gridSize, blockSize, sharedMemorySize >> > (d_input_array, d_output_array, arraySize);
		cudaEventRecord(stop);
		break;
	case 512:
		// Calculate the sum of the array
		cudaEventRecord(start);
		ReductionSumArrayKernel7<blockSize> << <gridSize, blockSize, sharedMemorySize >> > (d_input_array, d_output_array, arraySize);
		cudaEventRecord(stop);
		break;
	case 256:
		// Calculate the sum of the array
		cudaEventRecord(start);
		ReductionSumArrayKernel7<blockSize> << <gridSize, blockSize, sharedMemorySize >> > (d_input_array, d_output_array, arraySize);
		cudaEventRecord(stop);
		break;
	case 128:
		// Calculate the sum of the array
		cudaEventRecord(start);
		ReductionSumArrayKernel7<blockSize> << <gridSize, blockSize, sharedMemorySize >> > (d_input_array, d_output_array, arraySize);
		cudaEventRecord(stop);
		break;
	case 64:
		// Calculate the sum of the array
		cudaEventRecord(start);
		ReductionSumArrayKernel7<blockSize> << <gridSize, blockSize, sharedMemorySize >> > (d_input_array, d_output_array, arraySize);
		cudaEventRecord(stop);
		break;
	case 32:
		// Calculate the sum of the array
		cudaEventRecord(start);
		ReductionSumArrayKernel7<blockSize> << <gridSize, blockSize, sharedMemorySize >> > (d_input_array, d_output_array, arraySize);
		cudaEventRecord(stop);
		break;
	case 16:
		// Calculate the sum of the array
		cudaEventRecord(start);
		ReductionSumArrayKernel7<blockSize> << <gridSize, blockSize, sharedMemorySize >> > (d_input_array, d_output_array, arraySize);
		cudaEventRecord(stop);
		break;
	case 8:
		// Calculate the sum of the array
		cudaEventRecord(start);
		ReductionSumArrayKernel7<blockSize> << <gridSize, blockSize, sharedMemorySize >> > (d_input_array, d_output_array, arraySize);
		cudaEventRecord(stop);
		break;
	case 4:
		// Calculate the sum of the array
		cudaEventRecord(start);
		ReductionSumArrayKernel7<blockSize> << <gridSize, blockSize, sharedMemorySize >> > (d_input_array, d_output_array, arraySize);
		cudaEventRecord(stop);
		break;
	case 2:
		// Calculate the sum of the array
		cudaEventRecord(start);
		ReductionSumArrayKernel7<blockSize> << <gridSize, blockSize, sharedMemorySize >> > (d_input_array, d_output_array, arraySize);
		cudaEventRecord(stop);
		break;
	case 1:
		// Calculate the sum of the array
		cudaEventRecord(start);
		ReductionSumArrayKernel7<blockSize> << <gridSize, blockSize, sharedMemorySize >> > (d_input_array, d_output_array, arraySize);
		cudaEventRecord(stop);
		break;
	}
	
	// Wait for the event to complete
	cudaEventSynchronize(stop);
	
	// Calculate the time
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	// Copy the output array from the GPU
	cudaMemcpy(output_array, d_output_array, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

	// Print the results
	std::cout << "The sum of the array is: " << output_array[0] << std::endl;

	// Print the time
	std::cout << "The time to calculate the sum of the array is: " << milliseconds << " ms" << std::endl;

	// Free the memory
	delete[] input_array;
	delete[] output_array;
	cudaFree(d_input_array);
	cudaFree(d_output_array);

	// Reset the device
	cudaDeviceReset();
}