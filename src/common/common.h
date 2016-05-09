#ifndef HPC_BOOTCAMP_COMMON_H
#define HPC_BOOTCAMP_COMMON_H

#include <assert.h>
#include <sys/time.h>
#include <time.h>
#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#define CHECK_CUDA(call) { \
    const cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR @ %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
}

unsigned long long current_time_ns() {
#ifdef __MACH__
    struct timeval t;
    assert(gettimeofday(&t, NULL) == 0); 
    unsigned long long s = 1000000000ULL * (unsigned long long)t.tv_sec;
    return ((unsigned long long)t.tv_usec) * 1000ULL + s;
#else
    struct timespec t ={0,0};
    clock_gettime(CLOCK_MONOTONIC, &t);
    unsigned long long s = 1000000000ULL * (unsigned long long)t.tv_sec;
    return (((unsigned long long)t.tv_nsec)) + s;
#endif
}

#endif
