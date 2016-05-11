vecadd: vecadd.cu
	nvcc -O3 -arch=sm_20 $^ -o $@ -I../common
clean:
	rm -f vecadd
