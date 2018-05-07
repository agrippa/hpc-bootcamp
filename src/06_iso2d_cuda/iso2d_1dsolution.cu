/*
 * Copyright (c) 2012, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived 
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Paulius Micikevicius (pauliusm@nvidia.com)
 * Max Grossman (jmg3@rice.edu)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include "common.h"
#include "common2d.h"

#define BDIM   256

__global__ void fwd_kernel(TYPE *next, TYPE *curr, TYPE *vsq, TYPE *c_coeff,
        int nx, int ny, int dimx, int radius) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int y = tid / nx;
    int x = tid % nx;
    int this_offset = POINT_OFFSET(x, y, dimx, radius);

    TYPE temp = 2.0f * curr[this_offset] - next[this_offset];
    TYPE div = c_coeff[0] * curr[this_offset];
    for (int d = 1; d <= radius; d++) {
        int y_pos_offset = POINT_OFFSET(x, y + d, dimx, radius);
        int y_neg_offset = POINT_OFFSET(x, y - d, dimx, radius);
        int x_pos_offset = POINT_OFFSET(x + d, y, dimx, radius);
        int x_neg_offset = POINT_OFFSET(x - d, y, dimx, radius);
        div += c_coeff[d] * (curr[y_pos_offset] +
                curr[y_neg_offset] + curr[x_pos_offset] +
                curr[x_neg_offset]);
    }
    next[this_offset] = temp + div * vsq[this_offset];
}

int main( int argc, char *argv[] ) {
    config conf;
    setup_config(&conf, argc, argv);
    init_progress(conf.progress_width, conf.nsteps, conf.progress_disabled);

    if ((conf.ny * conf.nx) % BDIM != 0) {
        fprintf(stderr, "Invalid problem configuration, ny x nx must be an "
                "even multiple of %d\n", BDIM);
        return 1;
    }

    TYPE dx = 20.f;
    TYPE dt = 0.002f;

    // compute the pitch for perfect coalescing
    size_t dimx = conf.nx + 2*conf.radius;
    size_t dimy = conf.ny + 2*conf.radius;
    size_t nbytes = dimx * dimy * sizeof(TYPE);

    if (conf.verbose) {
        printf("x = %zu, y = %zu\n", dimx, dimy);
        printf("nsteps = %d\n", conf.nsteps);
        printf("radius = %d\n", conf.radius);
    }

    TYPE c_coeff[NUM_COEFF];
    TYPE *curr = (TYPE *)malloc(nbytes);
    TYPE *next = (TYPE *)malloc(nbytes);
    TYPE *vsq  = (TYPE *)malloc(nbytes);
    if (curr == NULL || next == NULL || vsq == NULL) {
        fprintf(stderr, "Allocations failed\n");
        return 1;
    }

    config_sources(&conf.srcs, &conf.nsrcs, conf.nx, conf.ny, conf.nsteps);
    TYPE **srcs = sample_sources(conf.srcs, conf.nsrcs, conf.nsteps, dt);

    init_data(curr, next, vsq, c_coeff, dimx, dimy, dx, dt);

    TYPE *d_curr, *d_next, *d_vsq, *d_c_coeff;
    CHECK(cudaMalloc((void **)&d_curr, nbytes));
    CHECK(cudaMalloc((void **)&d_next, nbytes));
    CHECK(cudaMalloc((void **)&d_vsq, nbytes));
    CHECK(cudaMalloc((void **)&d_c_coeff, NUM_COEFF * sizeof(TYPE)));

    dim3 block(BDIM);
    dim3 grid((conf.nx * conf.ny) / BDIM);

    double mem_start = seconds();

    CHECK(cudaMemcpy(d_curr, curr, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_next, next, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_vsq, vsq, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_c_coeff, c_coeff, NUM_COEFF * sizeof(TYPE),
                cudaMemcpyHostToDevice));
    double start = seconds();
    for (int step = 0; step < conf.nsteps; step++) {
        for (int src = 0; src < conf.nsrcs; src++) {
            if (conf.srcs[src].t > step) continue;
            int src_offset = POINT_OFFSET(conf.srcs[src].x, conf.srcs[src].y,
                    dimx, conf.radius);
            CHECK(cudaMemcpy(d_curr + src_offset, srcs[src] + step,
                        sizeof(TYPE), cudaMemcpyHostToDevice));
        }

        fwd_kernel<<<grid, block>>>(d_next, d_curr, d_vsq, d_c_coeff,
                conf.nx, conf.ny, dimx, conf.radius);
        TYPE *tmp = d_next;
        d_next = d_curr;
        d_curr = tmp;

        update_progress(step + 1);
    }
    CHECK(cudaDeviceSynchronize());
    double compute_s = seconds() - start;

    CHECK(cudaMemcpy(curr, d_curr, nbytes, cudaMemcpyDeviceToHost));
    double total_s = seconds() - mem_start;

    float point_rate = (float)conf.nx * conf.ny / (compute_s / conf.nsteps);
    printf("iso_r4_2x:   %8.10f s total, %8.10f s/step, %8.2f Mcells/s/step\n",
            total_s, compute_s / conf.nsteps, point_rate / 1000000.f);

    if (conf.save_text) {
        save_text(curr, dimx, dimy, conf.ny, conf.nx, "snap.text", conf.radius);
    }

    free(curr);
    free(next);
    free(vsq);
    for (int i = 0; i < conf.nsrcs; i++) {
        free(srcs[i]);
    }
    free(srcs);

    CHECK(cudaFree(d_curr));
    CHECK(cudaFree(d_next));
    CHECK(cudaFree(d_vsq));
    CHECK(cudaFree(d_c_coeff));

    return 0;
}
