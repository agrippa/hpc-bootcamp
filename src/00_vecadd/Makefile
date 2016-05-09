vecadd: vecadd.cu
	nvcc -arch=sm_20 $^ -o $@ -I../common

clean:
	rm -f vecadd
