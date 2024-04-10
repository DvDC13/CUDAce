NVCC=nvcc

NVCCFLAGS= -Xcompiler -fopenmp

TARGET=main

all: main

main: choose_device.cu
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $<

clean:
	rm -f $(TARGET)