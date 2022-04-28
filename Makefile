NVCC = nvcc
# NVCCFLAGS = -Xcompiler -fopenmp -g -O3 -G --ptxas-options=-v
NVCCFLAGS = -Xcompiler -fopenmp -g -O3 --ptxas-options=-v

all: fsm convert

.PHONY: clean

fsm: fsm.cu graph.cc gspan.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

convert: convert.cc graph.cc
	$(NVCC) $(NVCCFLAGS) -o $@ $^

clean:
	rm -f fsm convert
