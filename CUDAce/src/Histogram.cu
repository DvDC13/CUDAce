#include "Histogram.cuh"

#include "Error.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cstdlib>

/*
* Histogram kernel on the GPU.
*/
__global__ void HistogramKernel(unsigned int* histogram, unsigned int* data, int arraySize)
{
	// Get the index of the thread
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	// Allocate shared memory
	__shared__ unsigned int sharedHistogram[256];

	// Initialize the shared memory
	if (threadIdx.x < 256) sharedHistogram[threadIdx.x] = 0;

	// Synchronize all the threads
	__syncthreads();

	// Calculate the index of the data locally
	for (int i = gid; i < arraySize; i += blockDim.x * gridDim.x)
		atomicAdd(&sharedHistogram[data[i]], 1);

	// Synchronize all the threads
	__syncthreads();
	
	// Add the local histogram to the global histogram
	if (threadIdx.x < 256) atomicAdd(&histogram[threadIdx.x], sharedHistogram[threadIdx.x]);
}

/*
* This function is used to calculate the histogram of an array on the CPU.
*/
void HistogramCPU(unsigned int* cpu_histogram, unsigned int* cpu_data, int arraySize)
{
	// Iterate till arraySize
	for (int i = 0; i < arraySize; i++)
		// Increment the histogram
		cpu_histogram[cpu_data[i]]++;
}

/*
* This function is used to calculate the histogram of an array.
 */
void Histogram(int arraySize)
{
	// Create the input and output arrays
	unsigned int* cpu_input = new unsigned int[arraySize];
	unsigned int* cpu_output = new unsigned int[256];
	unsigned int* gpu_input;
	unsigned int* gpu_histogram;
	unsigned int* cpu_histogram = new unsigned int[256];
	
	// Initialize the cpu histogram to 0
	memset(cpu_histogram, 0, 256 * sizeof(unsigned int));

	// Create the events used to measure the time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Initialize the input array
	for (int i = 0; i < arraySize; i++)
		cpu_input[i] = rand() % 256;

	// Start the timer
	cudaEventRecord(start);

	// Calculate histogram on the CPU
	HistogramCPU(cpu_histogram, cpu_input, arraySize);

	// Stop the timer
	cudaEventRecord(stop);

	// Wait for the stop event to complete
	cudaEventSynchronize(stop);

	// Calculate the time on the CPU
	float millisecondsCPU = 0;
	cudaEventElapsedTime(&millisecondsCPU, start, stop);

	// Allocate the memory on the GPU
	cudaMalloc((void**)&gpu_input, arraySize * sizeof(unsigned int));
	cudaMalloc((void**)&gpu_histogram, 256 * sizeof(unsigned int));

	// Copy the input array to the GPU
	cudaMemcpy(gpu_input, cpu_input, arraySize * sizeof(unsigned int), cudaMemcpyHostToDevice);

	// Initialize the output array to 0
	cudaMemset(gpu_histogram, 0, 256 * sizeof(unsigned int));

	// Calculate the number of blocks and threads
	int blockSize = 512;
	int numBlocks = arraySize / blockSize;

	// Define the kernel launch configuration
	dim3 dimBlock(blockSize);
	dim3 dimGrid(numBlocks);
	
	// Start the timer
	cudaEventRecord(start);

	// Call the kernel
	HistogramKernel << <dimGrid, dimBlock >> > (gpu_histogram, gpu_input, arraySize);

	// Stop the timer
	cudaEventRecord(stop);

	// Wait for the kernel to finish
	cudaDeviceSynchronize();
	cudaEventSynchronize(stop);
	
	// Copy the output array to the CPU
	cudaMemcpy(cpu_output, gpu_histogram, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	// Calculate the time on the GPU
	float millisecondsGPU = 0;
	cudaEventElapsedTime(&millisecondsGPU, start, stop);

	// Print the results
	int j = 0;
	for (int i = 0; i < 256; i++)
	{
		std::cout << "Number of " << i << "s: " << cpu_output[i] << std::endl;
		j += cpu_output[i];
	}
	std::cout << "Total: " << j << std::endl;
	std::cout << "CPU time: " << millisecondsCPU << " ms" << std::endl;
	std::cout << "GPU time: " << millisecondsGPU << " ms" << std::endl;

	// Free the memory on the GPU
	cudaFree(gpu_input);
	cudaFree(gpu_histogram);

	// Free the memory on the CPU
	delete[] cpu_input;
	delete[] cpu_output;
	delete[] cpu_histogram;
}