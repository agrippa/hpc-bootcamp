#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __CUDACC__
int getNumCUDADevices() {
    return 0;
}
#else
int getNumCUDADevices() {
    int ndevices;
    CHECK(cudaGetDeviceCount(&ndevices));
    return ndevices;
}
#endif

#ifdef __cplusplus
}
#endif
