NVCC=nvcc

NVCCFLAGS= -Xcompiler -fopenmp -O3 -arch=sm_75

TARGET=main

all: main

main: decoupledLookBack_scan.cu
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $<

clean:
	rm -f $(TARGET)