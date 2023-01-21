#include "ElementWiseSquare.cuh"

#include "Error.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <immintrin.h>

#include <iostream>
#include <cstdlib>

/*
* This function is used to square the elements of an array on the GPU.
*/
__global__ void ElementWiseSquareKernel(float* input, float* output, int arraySize)
{
	// Get the index of the thread
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Check if the index is within the array
	if (gid >= arraySize) return;
	
	// Calculate the square of the input
	output[gid] = input[gid] * input[gid];
}

/*
* This function is used to square the elements of an array on the CPU.
*/
void ElementWiseSquareCPU(float* cpu_input, float* cpu_output, int arraySize)
{
	// Iterate till arraySize
	for (int i = 0; i < arraySize; i += 4)
	{
		// Check if there are enough elements left
		if (i + 4 > arraySize)
		{
			// Iterate till arraySize
			for (int j = i; j < arraySize; j++)
				// Square the element
				cpu_output[j] = cpu_input[j] * cpu_input[j];
			break;
		}

		/*
		* These lines of code are using the AVX2 instruction set 
		* to perform the element-wise square operation on the CPU.
		*/
		
		// Load 4 elements
		__m256 input = _mm256_load_ps(cpu_input + i);
		// Square the elements
		__m256 output = _mm256_mul_ps(input, input);
		// Store the elements
		_mm256_store_ps(cpu_output + i, output);
	}
}

/*
* This function is used to compare the results of the GPU and CPU implementations.
*/
void CheckOutput(float* cpu_output, float* gpu_output, int arraySize)
{
	// Iterate till arraySize
	for (int i = 0; i < arraySize; i++)
	{
		// If the values are not equal print the error
		if (cpu_output[i] != gpu_output[i])
		{
			std::cerr << "CPU and GPU output do not match!" << std::endl;
			std::cerr << "CPU output: " << cpu_output[i] << std::endl;
			std::cerr << "GPU output: " << gpu_output[i] << std::endl;
			std::cerr << "Index: " << i << std::endl;
			exit(EXIT_FAILURE);
		}
	}
	
	// If the values are equal print the success message
	std::cout << "CPU and GPU output match!" << std::endl;
}

/*
* This function is used to square the elements of an array.
* The array is first copied to the GPU and then the kernel is launched.
* The results are then copied back to the CPU and compared with the CPU implementation.
* The array is then freed.
*/
void ElementWiseSquare(int arraySize)
{
	// Allocate host memory for input and output arrays
	unsigned int byteSize = arraySize * sizeof(float);
	float* host_input = (float *)malloc(byteSize);
	float* host_output = (float*)malloc(byteSize);
	float* cpu_output = (float*)malloc(byteSize);

	// Initialize input array on host
	for (int i = 0; i < arraySize; i++)
		host_input[i] = (float)i;

	// Calculate optimal block size for kernel launch
	int blockSize = 0;
	int minGridSize = 0;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ElementWiseSquareKernel, 0, arraySize);
	blockSize = std::min(blockSize, arraySize);

	// Create events to measure CPU and GPU execution time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// Measure CPU execution time
	cudaEventRecord(start, 0);
	ElementWiseSquareCPU(host_input, cpu_output, arraySize);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float cpuTime = 0;
	cudaEventElapsedTime(&cpuTime, start, stop);
	std::cout << "CPU time: " << cpuTime << " ms" << std::endl;

	// Create CUDA stream for data transfer
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// Allocate device memory for input and output arrays
	float* device_input;
	float* device_output;
	size_t pitch;
	CUDA_CALL(cudaMallocPitch(&device_input, &pitch, byteSize, 1));
	CUDA_CALL(cudaMalloc((void**)&device_output, byteSize));

	// Copy input array from host to device memory
	CUDA_CALL(cudaMemcpy2DAsync(device_input, pitch, host_input, byteSize, byteSize, 1, cudaMemcpyHostToDevice, stream));
	
	// Calculate grid size for kernel launch
	int gridSize = (arraySize + blockSize - 1) / blockSize;

	// Measure GPU execution time
	cudaEventRecord(start, 0);
	ElementWiseSquareKernel << <gridSize, blockSize, 0, stream>> > (device_input, device_output, arraySize);
	cudaEventRecord(stop, 0);
	CUDA_CALL(cudaDeviceSynchronize());
	cudaEventSynchronize(stop);
	float gpuTime = 0;
	cudaEventElapsedTime(&gpuTime, start, stop);
	std::cout << "GPU time: " << gpuTime << " ms" << std::endl;

	// Copy output array from device to host memory
	CUDA_CALL(cudaMemcpyAsync(host_output, device_output, byteSize, cudaMemcpyDeviceToHost, stream));
	CUDA_CALL(cudaStreamSynchronize(stream));
	
	// Compare CPU and GPU output to ensure they match
	CheckOutput(cpu_output, host_output, arraySize);
	
	// Free allocated host and device memory
	free(host_input);
	free(host_output);
	CUDA_CALL(cudaFree(device_input));
	CUDA_CALL(cudaFree(device_output));
	cudaStreamDestroy(stream);

	// Reset CUDA device
	cudaDeviceReset();
}