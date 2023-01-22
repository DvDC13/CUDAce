#pragma once

/*
* ElementWiseSquare.cuh
* This file contains the declarations of the functions used to square the elements of an array.
* The elements are squared on the GPU using the CUDA programming model.
* The elements are squared on the CPU using the AVX2 instruction set.
*/
void ElementWiseSquare(int arraySize);