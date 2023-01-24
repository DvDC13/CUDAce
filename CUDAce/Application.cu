#include <iostream>

#include "ElementWiseSquare.cuh"
#include "MatrixMultiplication.cuh"
#include "Histogram.cuh"

int main(int argc, char** argv)
{
	//unsigned int arraySize = atoi(argv[1]);
	//ElementWiseSquare(arraySize);

	/*const int row1 = atoi(argv[1]);
	const int col1 = atoi(argv[2]);
	const int row2 = atoi(argv[3]);
	const int col2 = atoi(argv[4]);
	
	MatrixMultiplication(row1, col1, row2, col2);
	*/

	unsigned int arraySize = atoi(argv[1]);
	Histogram(arraySize);
	
	return 0;
}