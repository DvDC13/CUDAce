#include "MatrixMultiplication.cuh"

#include "Error.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cstdlib>

/*
* Matrix multiplication kernel on the GPU.
* Each thread computes one element of the result matrix.
*/
__global__ void MatrixMultiplicationKernel(float* Matrix1, float* Matrix2, float* MatrixResult, int row1, int col1, int row2, int col2)
{
	// Get the index of the thread
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Check if the index is within the array
	if (row >= row1 || col >= col2) return;

	// Calculate the element of the output matrix
	float sum = 0.0f;
	for (int k = 0; k < row2; k++)
		sum += Matrix1[row * col1 + k] * Matrix2[k * col2 + col];
	MatrixResult[row * col2 + col] = sum;
}

/*
* This function is used to multiply two matrices on the CPU.
*/
void MatrixMultiplicationCPU(float* Matrix1, float* Matrix2, float* MatrixResult, int row1, int col1, int row2, int col2)
{
	// Iterate till row1
	for (int i = 0; i < row1; i++)
	{
		// Iterate till col2
		for (int j = 0; j < col2; j++)
		{
			// Calculate the element of the output matrix
			float sum = 0.0f;
			for (int k = 0; k < col1; k++)
				sum += Matrix1[i * col1 + k] * Matrix2[k * col2 + j];
			MatrixResult[i * col2 + j] = sum;
		}
	}
}

/*
* This function is used to compare the results of the GPU and CPU implementations.
*/
void CheckMatrixMultiplication(float* MatrixCpu, float* MatrixGpu, int row1, int col2)
{
	// Iterate till row1
	for (int i = 0; i < row1; i++)
	{
		// Iterate till col2
		for (int j = 0; j < col2; j++)
		{
			// Check if the element is the same
			if (MatrixCpu[i * col2 + j] != MatrixGpu[i * col2 + j])
			{
				std::cerr << "Error: The results of the GPU and CPU implementations do not match." << std::endl;
				std::cerr << "MatrixCpu[" << i << "][" << j << "] = " << MatrixCpu[i * col2 + j] << std::endl;
				std::cerr << "MatrixGpu[" << i << "][" << j << "] = " << MatrixGpu[i * col2 + j] << std::endl;
				exit(EXIT_FAILURE);
			}
		}
	}
	
	// Print the result of the check
	std::cout << "The results of the GPU and CPU implementations match." << std::endl;
}

/*
* This function is used to initialize the matrices.
*/
void InitializeMatrix(float* Matrix, int row, int col)
{
	// Iterate till row
	for (int i = 0; i < row; i++)
		// Iterate till col
		for (int j = 0; j < col; j++)
			// Initialize the element
			Matrix[i * col + j] = (float)rand() / (float)RAND_MAX;
}

/*
* This function is used to print the matrices.
*/
void PrintMatrix(float* Matrix, int row, int col)
{
	// Iterate till row
	for (int i = 0; i < row; i++)
	{
		// Iterate till col
		for (int j = 0; j < col; j++)
			// Print the element
			std::cout << Matrix[i * col + j] << " ";
		// Print a new line
		std::cout << std::endl;
	}
	// Print a new line
	std::cout << std::endl;
}

/*
* This function is used to multiply two matrices on the GPU.
* The matrices are copied to the GPU, the kernel is executed and the result is copied back to the CPU.
*/
void MatrixMultiplication(const int Matrix1Row, const int Matrix1Col, const int Matrix2Row, const int Matrix2Col)
{
	// Check if the matrices can be multiplied
	if (Matrix1Col != Matrix2Row)
	{
		std::cerr << "The number of columns in the first matrix must be equal to the number of rows in the second matrix." << std::endl;
		exit(EXIT_FAILURE);
	}

	// Create events to measure CPU and GPU execution time
	cudaEvent_t startCpu, stopCpu, startGpu, stopGpu;
	cudaEventCreate(&startCpu);
	cudaEventCreate(&stopCpu);
	cudaEventCreate(&startGpu);
	cudaEventCreate(&stopGpu);

	// Allocate host matrices
	float* host_Matrix1 = new float[Matrix1Row * Matrix1Col];
	float* host_Matrix2 = new float[Matrix2Row * Matrix2Col];
	float* host_MatrixResult = new float[Matrix1Row * Matrix2Col];
	float* host_MatrixResultCpu = new float[Matrix1Row * Matrix2Col];

	// Initialize host matrices
	InitializeMatrix(host_Matrix1, Matrix1Row, Matrix1Col);
	InitializeMatrix(host_Matrix2, Matrix2Row, Matrix2Col);

	// Measure CPU execution time
	cudaEventRecord(startCpu, 0);
	
	// Calculate host matrix in cpu
	MatrixMultiplicationCPU(host_Matrix1, host_Matrix2, host_MatrixResultCpu, Matrix1Row, Matrix1Col, Matrix2Row, Matrix2Col);
	
	cudaEventRecord(stopCpu, 0);
	cudaEventSynchronize(stopCpu);
	float cpuTime;
	cudaEventElapsedTime(&cpuTime, startCpu, stopCpu);
	
	std::cout << "Matrix 1:" << std::endl;
	PrintMatrix(host_Matrix1, Matrix1Row, Matrix1Col);
	std::cout << "Matrix 2:" << std::endl;
	PrintMatrix(host_Matrix2, Matrix2Row, Matrix2Col);
	std::cout << "Matrix Result CPU:" << std::endl;
	PrintMatrix(host_MatrixResultCpu, Matrix1Row, Matrix2Col);

	// Allocate device matrices
	float* device_Matrix1 = nullptr;
	float* device_Matrix2 = nullptr;
	float* device_MatrixResult = nullptr;
	
	// Allocate memory for the matrices on the device
	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&device_Matrix1), Matrix1Row * Matrix1Col * sizeof(float)));
	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&device_Matrix2), Matrix2Row * Matrix2Col * sizeof(float)));
	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&device_MatrixResult), Matrix1Row * Matrix2Col * sizeof(float)));

	// Copy the matrices from the host to the device
	CUDA_CALL(cudaMemcpy(device_Matrix1, host_Matrix1, Matrix1Row * Matrix1Col * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(device_Matrix2, host_Matrix2, Matrix2Row * Matrix2Col * sizeof(float), cudaMemcpyHostToDevice));

	// Define block size
	dim3 blockDim(16, 16, 1);

	// Calculate grid size components based on the number of rows and columns of the matrices
	int gridSizeX = (Matrix1Row + blockDim.x - 1) / blockDim.x;
	int gridSizeY = (Matrix2Col + blockDim.y - 1) / blockDim.y;

	// Define grid size
	dim3 gridDim(gridSizeX, gridSizeY, 1);

	// Measure GPU execution time
	cudaEventRecord(startGpu, 0);
	
	// Launch the kernel
	MatrixMultiplicationKernel << <gridDim, blockDim >> > (device_Matrix1, device_Matrix2, device_MatrixResult, Matrix1Row, Matrix1Col, Matrix2Row, Matrix2Col);
	
	cudaEventRecord(stopGpu, 0);
	cudaEventSynchronize(stopGpu);
	CUDA_CALL(cudaDeviceSynchronize());

	float gpuTime;
	cudaEventElapsedTime(&gpuTime, startGpu, stopGpu);

	// Copy the result from the device to the host
	CUDA_CALL(cudaMemcpy(host_MatrixResult, device_MatrixResult, Matrix1Row * Matrix2Col * sizeof(float), cudaMemcpyDeviceToHost));
	
	// print the result
	std::cout << "Matrix Result GPU:" << std::endl;
	PrintMatrix(host_MatrixResult, Matrix1Row, Matrix2Col);
	
	// Check the result
	CheckMatrixMultiplication(host_MatrixResultCpu, host_MatrixResult, Matrix1Row, Matrix2Col);

	// Print the execution time
	std::cout << "CPU execution time: " << cpuTime << " ms" << std::endl;
	std::cout << "GPU execution time: " << gpuTime << " ms" << std::endl;
	
	// Deallocate the host matrices
	delete[] host_Matrix1;
	delete[] host_Matrix2;
	delete[] host_MatrixResult;
	delete[] host_MatrixResultCpu;

	// Deallocate the device matrices
	CUDA_CALL(cudaFree(device_Matrix1));
	CUDA_CALL(cudaFree(device_Matrix2));
	CUDA_CALL(cudaFree(device_MatrixResult));

	// Reset the device
	CUDA_CALL(cudaDeviceReset());
}