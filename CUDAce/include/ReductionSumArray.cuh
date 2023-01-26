#pragma once

/*
* ReductionSumArray.cuh
* This file contains the declarations of the functions used to sum the elements of an array.
* The array is summed on the GPU using the CUDA programming model.
* The sum is performed using a reduction algorithm.
* There is 7 methods to sum the elements of an array.
* "https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf" is the source of the reduction algorithm.
*/
void ReductionSumArray(int arraySize);