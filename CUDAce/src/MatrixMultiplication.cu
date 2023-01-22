#include "MatrixMultiplication.cuh"

#include "Error.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cstdlib>

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

void CheckMatrixMultiplication(float* MatrixCpu, float* MatrixGpu, int row1, int col2)
{
	for (int i = 0; i < row1; i++)
	{
		for (int j = 0; j < col2; j++)
		{
			if (MatrixCpu[i * col2 + j] != MatrixGpu[i * col2 + j])
			{
				std::cerr << "Error: The results of the GPU and CPU implementations do not match." << std::endl;
				std::cerr << "MatrixCpu[" << i << "][" << j << "] = " << MatrixCpu[i * col2 + j] << std::endl;
				std::cerr << "MatrixGpu[" << i << "][" << j << "] = " << MatrixGpu[i * col2 + j] << std::endl;
				exit(EXIT_FAILURE);
			}
		}
	}
	
	std::cout << "The results of the GPU and CPU implementations match." << std::endl;
}

void InitializeMatrix(float* Matrix, int row, int col)
{
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			Matrix[i * col + j] = (float)rand() / (float)RAND_MAX;
}

void PrintMatrix(float* Matrix, int row, int col)
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
			std::cout << Matrix[i * col + j] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void MatrixMultiplication(const int Matrix1Row, const int Matrix1Col, const int Matrix2Row, const int Matrix2Col)
{
	if (Matrix1Col != Matrix2Row)
	{
		std::cerr << "The number of columns in the first matrix must be equal to the number of rows in the second matrix." << std::endl;
		exit(EXIT_FAILURE);
	}

	// Allocate host matrices
	float* host_Matrix1 = new float[Matrix1Row * Matrix1Col];
	float* host_Matrix2 = new float[Matrix2Row * Matrix2Col];
	float* host_MatrixResult = new float[Matrix1Row * Matrix2Col];
	float* host_MatrixResultCpu = new float[Matrix1Row * Matrix2Col];

	// Initialize host matrices
	InitializeMatrix(host_Matrix1, Matrix1Row, Matrix1Col);
	InitializeMatrix(host_Matrix2, Matrix2Row, Matrix2Col);

	// Calculate host matrix in cpu
	MatrixMultiplicationCPU(host_Matrix1, host_Matrix2, host_MatrixResultCpu, Matrix1Row, Matrix1Col, Matrix2Row, Matrix2Col);

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
	
	// Calculate the optimal block size
	int blockSize = 0;
	int minGridSize = 0;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, MatrixMultiplicationKernel, 0, Matrix1Row * Matrix2Col);
	blockSize = std::min(blockSize, Matrix1Row * Matrix2Col);

	// Calculate the optimal grid size
	int gridSize = (Matrix1Row * Matrix2Col + blockSize - 1) / blockSize;

	// Calculate the dim3 grid and block size
	dim3 dimBlock(blockSize, blockSize, 1);
	dim3 dimGrid(gridSize, 1, 1);

	// Launch the kernel
	MatrixMultiplicationKernel << <dimGrid, dimBlock >> > (device_Matrix1, device_Matrix2, device_MatrixResult, Matrix1Row, Matrix1Col, Matrix2Row, Matrix2Col);
	CUDA_CALL(cudaDeviceSynchronize());

	// Copy the result from the device to the host
	CUDA_CALL(cudaMemcpy(host_MatrixResult, device_MatrixResult, Matrix1Row * Matrix2Col * sizeof(float), cudaMemcpyDeviceToHost));
	
	// print the result
	std::cout << "Matrix Result GPU:" << std::endl;
	PrintMatrix(host_MatrixResult, Matrix1Row, Matrix2Col);
	
	// Check the result
	CheckMatrixMultiplication(host_MatrixResultCpu, host_MatrixResult, Matrix1Row, Matrix2Col);

	//deallocate the host matrices
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
	
	exit(EXIT_SUCCESS);
}