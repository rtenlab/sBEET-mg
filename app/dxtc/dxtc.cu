#include <sys/types.h>
#include <sys/fcntl.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

#include "dxtc_kernel.cuh"
#include "dxtc.h"
#include "common/include/STGM.h"
#include "common/include/messages.h"

#include "CudaMath.h"
#include "dds.h"
#include "permutations.h"
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <float.h> // for FLT_MAX


#define ERROR_THRESHOLD 0.02f

// #define __debugsync()

/* 
    Take the input arguments for this class
*/
DXTC_TASK::DXTC_TASK(std::string fn) {
    input_image = fn;
    const char *image_path = input_image.c_str();
    if (!sdkLoadPPM4ub(image_path, &data, &w, &h))
        fprintf(stderr, "Error, unable to open source image file <%s>\n", image_path);

    memSize = w * h * 4;
    compressedSize = (w / 4) * (h / 4) * 8;
};

/*  */
DXTC_TASK::~DXTC_TASK() {};

/* 
    Init: 
    1, Malloc (host, device)
    2. definition of variables if necessary
*/
void DXTC_TASK::taskInit() {
    // Allocate input image.
    block_image = (unsigned int *)malloc(memSize);
    // Convert linear image to block linear.
    for (int by = 0; by < h/4; by++) {
        for (int bx = 0; bx < w/4; bx++) {
            for (int i = 0; i < 16; i++) {
                const int x = i & 3;
                const int y = i / 4;
                block_image[(by * w/4 + bx) * 16 + i] = ((unsigned int *)data)[(by * 4 + y) * 4 * (w/4) + bx * 4 + x];
            }
        }
    }

    computePermutations(permutations);

    // h_result = (unsigned int *)malloc(compressedSize);
};

void DXTC_TASK::taskInitDevice(unsigned int _id) {
    STGM_INIT(_id);
    cudaMallocHost((void**)&plan[_id].h_result, compressedSize);

    cudaMalloc((void **) &plan[_id].d_data, memSize);
    cudaMalloc((void **)&plan[_id].d_result, compressedSize);
    cudaMalloc((void **) &plan[_id].d_permutations, 1024 * sizeof(unsigned int));

    cudaDeviceSynchronize();
}

/* 
    Periodic CPU works PRE kernel launching
    1. fresh some values
    2. memcpy, async on a specific stream
*/
void DXTC_TASK::taskPreGPU(cudaStream_t* s, unsigned int _id) {
    the_stream = s;
    cudaMemcpyAsync(plan[_id].d_permutations, permutations, 1024 * sizeof(unsigned int), cudaMemcpyHostToDevice, *the_stream);
    cudaMemcpyAsync(plan[_id].d_data, block_image, memSize, cudaMemcpyHostToDevice, *the_stream);
}

/*  */
void DXTC_TASK::taskPostGPU(unsigned int _id) {
    cudaMemcpyAsync(plan[_id].h_result, plan[_id].d_result, compressedSize, cudaMemcpyDeviceToHost, *the_stream);

    // char output_filename[1024];
    // strcpy(output_filename, image_path);
    // strcpy(output_filename + strlen(image_path) - 3, "dds");

    // const char *image_path = input_image.c_str();
    // char* output_filename = (char*)image_path;
    // strcpy(output_filename + strlen(image_path) - 3, "dds");

    // FILE *fp = fopen(output_filename, "wb");
    // if (fp == 0) {
    //     printf("Error, unable to open output image <%s>\n", output_filename);
    //     exit(EXIT_FAILURE);
    // }

    // DDSHeader header;
    // header.fourcc = FOURCC_DDS;
    // header.size = 124;
    // header.flags  = (DDSD_WIDTH|DDSD_HEIGHT|DDSD_CAPS|DDSD_PIXELFORMAT|DDSD_LINEARSIZE);
    // header.height = h;
    // header.width = w;
    // header.pitch = compressedSize;
    // header.depth = 0;
    // header.mipmapcount = 0;
    // memset(header.reserved, 0, sizeof(header.reserved));
    // header.pf.size = 32;
    // header.pf.flags = DDPF_FOURCC;
    // header.pf.fourcc = FOURCC_DXT1;
    // header.pf.bitcount = 0;
    // header.pf.rmask = 0;
    // header.pf.gmask = 0;
    // header.pf.bmask = 0;
    // header.pf.amask = 0;
    // header.caps.caps1 = DDSCAPS_TEXTURE;
    // header.caps.caps2 = 0;
    // header.caps.caps3 = 0;
    // header.caps.caps4 = 0;
    // header.notused = 0;
    // fwrite(&header, sizeof(DDSHeader), 1, fp);
    // fwrite(h_result, compressedSize, 1, fp);
    // fclose(fp);
}

/* 
    Launch kernel:
    1. SM allocation, decided by the algorithm
    2. launch the kernel(s)
*/
void DXTC_TASK::taskRunOnGPU(unsigned int _id, std::vector<int> &_sm_arr) {
	// int sm_num = _sm_arr.size();
	int __n = _sm_arr.size();
    for (int i = 0; i < plan[_id].sm_num - __n; i++) {
        _sm_arr.push_back(-1);
    }
    STGM_SM_MAPPING(plan[_id].sm_num, _sm_arr);

    //int numIterations = 1;
    int numIterations = 3;
    unsigned int blocks = ((w + 3) / 4) * ((h + 3) / 4);
    int blocksPerLaunch = min(blocks, 768 * plan[_id].sm_num);
    for (int i = -1; i < numIterations; ++i) {
        for (int j = 0; j < (int)blocks; j += blocksPerLaunch) {
            STGM_LAUNCH_KERNEL(_id, compress, *the_stream, min(blocksPerLaunch, blocks-j), NUM_THREADS, plan[_id].d_permutations, plan[_id].d_data, (uint2 *)plan[_id].d_result, j);
        }
    }
}

/*  */
void DXTC_TASK::taskFinish() {
    for (int _id = 0; _id < 2; _id++) {
        cudaFree(plan[_id].d_permutations);
        cudaFree(plan[_id].d_data);
        cudaFree(plan[_id].d_result);
        cudaFreeHost(plan[_id].h_result);
        STGM_FINISH(_id);
    }
    free(block_image);
    // free(image_path);
    free(data);
}
