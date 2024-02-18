NVCC=nvcc

NVCCFLAGS= -Xcompiler -fopenmp

TARGET=main

all: main

main: sinus_sum.cu
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $<

clean:
	rm -f $(TARGET)