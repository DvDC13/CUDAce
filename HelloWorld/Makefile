NVCC=nvcc

NVCCFLAGS= -Xcompiler -fopenmp

TARGET=main

all: main

main: hello_world.cu
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $<

clean:
	rm -f $(TARGET)