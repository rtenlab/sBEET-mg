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
// FIXME:
// #include <errno.h>
// #include <chrono>
// #include <mutex>
// #include <thread>

#include <helper_functions.h>  // helper for shared that are common to CUDA Samples
#include <helper_cuda.h>       // helper for checking cuda initialization and error checking
#include <helper_string.h>     // helper functions for string parsing

#include "stereodisparity_kernel.cuh"
#include "stereodisparity.h"
#include "common/include/STGM.h"
#include "common/include/messages.h"


/* 
    Take the input arguments for this class
*/
STEREODISPARITY_TASK::STEREODISPARITY_TASK() {
    minDisp = -16;
    maxDisp = 0;
    // Load image data
    //allocate mem for the images on host side
    //initialize pointers to NULL to request lib call to allocate as needed
    // PPM images are loaded into 4 byte/pixel memory (RGBX)
    // char *fname0 = sdkFindFilePath("/home/wyd/Documents/rten/gpu_energy/app/stereodisparity/stereo.im0.640x533.ppm", "/home/wyd/Documents/rten/gpu_energy/app/stereodisparity/");
    // char *fname1 = sdkFindFilePath("app/stereodisparity/stereo.im1.640x533.ppm", "/home/wyd/Documents/rten/gpu_energy/app/stereodisparity/");
    std::string f1 = "app/stereodisparity/data/stereo.im0.640x533.ppm";
    std::string f2 = "app/stereodisparity/data/stereo.im1.640x533.ppm";
    const char *fname0 = f1.c_str();
    const char *fname1 = f2.c_str();
    if (!sdkLoadPPM4ub(fname0, &h_img0, &w, &h)) {
        fprintf(stderr, "Failed to load <%s>\n", fname0);
    }
    if (!sdkLoadPPM4ub(fname1, &h_img1, &w, &h)) {
        fprintf(stderr, "Failed to load <%s>\n", fname1);
    }
    
    DimBlock = dim3(blockSize_x, blockSize_y, 1);
    DimGrid = dim3(iDivUp(w, DimBlock.x), iDivUp(h, DimBlock.y));
    numData = w*h;
    memSize = sizeof(int) * numData;
};

/*  */
STEREODISPARITY_TASK::~STEREODISPARITY_TASK() {};

/* 
    Init: 
    1, Malloc (host, device)
    2. definition of variables if necessary
*/
void STEREODISPARITY_TASK::taskInit() {
    // WORKLOAD_INFO("STEREODISPARITY", "Allocating varribles...");
    // h_odata = (unsigned int *)malloc(memSize);    
};

void STEREODISPARITY_TASK::taskInitDevice(unsigned int _id) {
    STGM_INIT(_id);
    //allocate mem for the result on host side
    cudaMallocHost((void**)&plan[_id].h_odata, memSize);
    //initialize the memory
    for (unsigned int i = 0; i < numData; i++) plan[_id].h_odata[i] = 0;
    // allocate device memory for result
    cudaMalloc((void **) &plan[_id].d_odata, memSize);
    cudaMalloc((void **) &plan[_id].d_img0, memSize);
    cudaMalloc((void **) &plan[_id].d_img1, memSize);

    cudaDeviceSynchronize();
}

/* 
    Periodic CPU works PRE kernel launching
    1. fresh value of input variables to the kernel
    2. memcpy, async on a specific stream
*/
void STEREODISPARITY_TASK::taskPreGPU(cudaStream_t* s, unsigned int _id) {
    // WORKLOAD_INFO("STEREODISPARITY", "Memory copying HtoD...");
    the_stream = s;
    
    cudaMemcpyAsync(plan[_id].d_img0,  h_img0, memSize, cudaMemcpyHostToDevice, *the_stream);
    cudaMemcpyAsync(plan[_id].d_img1,  h_img1, memSize, cudaMemcpyHostToDevice, *the_stream);
    cudaMemcpyAsync(plan[_id].d_odata, plan[_id].h_odata, memSize, cudaMemcpyHostToDevice, *the_stream);

    size_t offset = 0;
    cudaChannelFormatDesc ca_desc0 = cudaCreateChannelDesc<unsigned int>();
    cudaChannelFormatDesc ca_desc1 = cudaCreateChannelDesc<unsigned int>();

    tex2Dleft.addressMode[0] = cudaAddressModeClamp;
    tex2Dleft.addressMode[1] = cudaAddressModeClamp;
    tex2Dleft.filterMode     = cudaFilterModePoint;
    tex2Dleft.normalized     = false;
    tex2Dright.addressMode[0] = cudaAddressModeClamp;
    tex2Dright.addressMode[1] = cudaAddressModeClamp;
    tex2Dright.filterMode     = cudaFilterModePoint;
    tex2Dright.normalized     = false;
    cudaBindTexture2D(&offset, tex2Dleft,  plan[_id].d_img0, ca_desc0, w, h, w*4);
    assert(offset == 0);
    cudaBindTexture2D(&offset, tex2Dright, plan[_id].d_img1, ca_desc1, w, h, w*4);
    assert(offset == 0);
}

/* 
    1. memcpy back
    2. some post-processing
*/
void STEREODISPARITY_TASK::taskPostGPU(unsigned int _id) {
    // WORKLOAD_INFO("STEREODISPARITY", "Memory copying DtoH...");
    cudaMemcpyAsync(plan[_id].h_odata, plan[_id].d_odata, memSize, cudaMemcpyDeviceToHost, *the_stream);
}

/* 
    Launch kernel:
    1. SM allocation, decided by the algorithm
    2. launch the kernel(s)
*/
void STEREODISPARITY_TASK::taskRunOnGPU(unsigned int _id, std::vector<int> &_sm_arr) {
	// int sm_num = _sm_arr.size();
	int __n = _sm_arr.size();
    for (int i = 0; i < plan[_id].sm_num - __n; i++) {
        _sm_arr.push_back(-1);
    }
    STGM_SM_MAPPING(plan[_id].sm_num, _sm_arr);
    // WORKLOAD_INFO("STEREODISPARITY", "Launching kernel...");
    STGM_LAUNCH_KERNEL(_id, stereoDisparityKernel, *the_stream, DimGrid, DimBlock, plan[_id].d_img0, plan[_id].d_img1, plan[_id].d_odata, w, h, minDisp, maxDisp);
}

/*  */
void STEREODISPARITY_TASK::taskFinish() {
    // cleanup memory
    for (int _id = 0; _id < 2; _id++) {
        cudaFree(plan[_id].d_odata);
        cudaFree(plan[_id].d_img0);
        cudaFree(plan[_id].d_img1);
        cudaFreeHost(plan[_id].h_odata);

        STGM_FINISH(_id);
    }
    
    free(h_img0);
    free(h_img1);

    // cudaDestroyTextureObject(ca_desc0);
    // cudaDestroyTextureObject(ca_desc1);
}

int STEREODISPARITY_TASK::iDivUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}
