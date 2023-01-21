#include <iostream>

#include "ElementWiseSquare.cuh"

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		// TODO: Print usage
		std::cerr << std::endl;
		exit(EXIT_FAILURE);
	}
	
	unsigned int arraySize = atoi(argv[1]);
	ElementWiseSquare(arraySize);
	
	return 0;
}