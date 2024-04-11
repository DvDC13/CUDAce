NVCC=nvcc

NVCCFLAGS= -Xcompiler -fopenmp

TARGET=main

all: main

main: julia_set.cu
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $<

clean:
	rm -f $(TARGET) julia.ppm