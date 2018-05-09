iso2d_cuda: iso2d.cu ../iso2d_common/common.cpp ../iso2d_common/common2d.cpp ../iso2d_common/common_cuda.cpp
	nvcc -O3 -arch=sm_20 $^ -o $@ -I../iso2d_common

clean:
	rm -f iso2d_cuda
