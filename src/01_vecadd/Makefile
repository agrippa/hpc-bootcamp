vecadd: vecadd.cu
	nvcc -arch=sm_20 $^ -o $@ -I../common -O3

clean:
	rm -f vecadd
