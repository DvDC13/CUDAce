NVCC=nvcc

NVCCFLAGS= -Xcompiler -fopenmp

TARGET=main

all: main

main: vector_sum.cu
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $<

clean:
	rm -f $(TARGET)