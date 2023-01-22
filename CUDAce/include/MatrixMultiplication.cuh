#pragma once

/*
* MatrixMultiplication.cuh
* This file contains the declarations of the functions used to multiply two matrices.
* The matrices are multiplied on the GPU using the CUDA programming model.
*/
void MatrixMultiplication(const int Matrix1Row, const int Matrix1Col, const int Matrix2Row, const int Matrix2Col);