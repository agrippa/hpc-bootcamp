all: iso2d_seq iso2d_omp

iso2d_seq: iso2d.cpp ../iso2d_common/common.cpp ../iso2d_common/common2d.cpp ../iso2d_common/common_cuda.cpp
	nvcc -O3 $^ -o $@ -I../iso2d_common

iso2d_omp: iso2d.cpp ../iso2d_common/common.cpp ../iso2d_common/common2d.cpp ../iso2d_common/common_cuda.cpp
	nvcc -O3 $^ -o $@ -I../iso2d_common -Xcompiler -fopenmp

clean:
	rm -f iso2d_seq iso2d_omp

