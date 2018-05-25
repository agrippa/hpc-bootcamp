cublas: cublas.cu
	nvcc -O3 -arch=sm_20 $^ -o $@ -lcublas

cublas_solution: cublas_solution.cu
	nvcc -O3 -arch=sm_20 $^ -o $@ -lcublas

clean:
	rm -f cublas cublas_solution
