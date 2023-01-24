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

	// Check if the index is within the array
	if (gid >= arraySize) return;

	// Get the value of the element
	unsigned int value = data[gid];

	// Increment the histogram
	atomicAdd(&histogram[value], 1);
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

	// Initialize the input array
	for (int i = 0; i < arraySize; i++)
		cpu_input[i] = rand() % 256;

	// Allocate the memory on the GPU
	cudaMalloc((void**)&gpu_input, arraySize * sizeof(unsigned int));
	cudaMalloc((void**)&gpu_histogram, 256 * sizeof(unsigned int));

	// Copy the input array to the GPU
	cudaMemcpy(gpu_input, cpu_input, arraySize * sizeof(unsigned int), cudaMemcpyHostToDevice);

	// Initialize the output array to 0
	cudaMemset(gpu_histogram, 0, 256 * sizeof(unsigned int));

	// Calculate the number of blocks and threads
	int blockSize = 32;
	int numBlocks = (arraySize + blockSize - 1) / blockSize;

	// Define the kernel launch configuration
	dim3 dimBlock(blockSize);
	dim3 dimGrid(numBlocks);
	
	// Call the kernel
	HistogramKernel << <dimGrid, dimBlock >> > (gpu_histogram, gpu_input, arraySize);

	// Copy the output array to the CPU
	cudaMemcpy(cpu_output, gpu_histogram, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	// Print the results
	int j = 0;
	for (int i = 0; i < 256; i++)
	{
		std::cout << "Number of " << i << "s: " << cpu_output[i] << std::endl;
		j += cpu_output[i];
	}
	std::cout << "Total: " << j << std::endl;

	// Free the memory on the GPU
	cudaFree(gpu_input);
	cudaFree(gpu_histogram);

	// Free the memory on the CPU
	delete[] cpu_input;
	delete[] cpu_output;
}