vecadd: vecadd.cu
	nvcc -arch=sm_20 $^ -o $@ -I../common -O3

vecadd_solution: vecadd_solution.cu
	nvcc -arch=sm_20 $^ -o $@ -I../common -O3

clean:
	rm -f vecadd
