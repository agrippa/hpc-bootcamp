cublas: cublas.cu
	nvcc -O3 -arch=sm_20 $^ -o $@

cublas_solution: cublas_solution.cu
	nvcc -O3 -arch=sm_20 $^ -o $@

clean:
	rm -f cublas cublas_solution
