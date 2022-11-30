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
#include <helper_cuda.h>
// FIXME:
// #include <errno.h>
// #include <chrono>
// #include <mutex>
// #include <thread>

#include "mmul_kernel.cuh"
#include "mmul.h"
#include "common/include/STGM.h"
#include "common/include/messages.h"

/* 
    Take the input arguments for this class
*/
MMUL_TASK::MMUL_TASK(int M, int N, int K) {
    matArow = M;
    matAcol = matBrow = N;
    matBcol = K;

    A_sz = matArow * matAcol;
    B_sz = matBrow * matBcol;
    C_sz = matArow * matBcol;

    DimGrid = dim3((matArow - 1) / BLOCK_SIZE + 1, (matBcol - 1) / BLOCK_SIZE + 1);
    DimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
};

/*  */
MMUL_TASK::~MMUL_TASK() {};

/* 
    Init: 
    1, Malloc (host, device)
    2. definition of variables if necessary
*/
void MMUL_TASK::taskInit() {
    // WORKLOAD_INFO("MMUL", "Allocating varribles...");
    A_h = (float*) malloc( sizeof(float) * A_sz );
    B_h = (float*) malloc( sizeof(float) * B_sz );
    for (unsigned int i=0; i < A_sz; i++) { A_h[i] = (rand()%100)/100.00; }
    for (unsigned int i=0; i < B_sz; i++) { B_h[i] = (rand()%100)/100.00; }
};

/* 
 * Need `gpu_num` copies of calling
 */
void MMUL_TASK::taskInitDevice(unsigned int _id) {
    STGM_INIT(_id);
    cudaMallocHost((void**)&plan[_id].C_h, sizeof(float) * C_sz );

    cudaMalloc( (void**)&plan[_id].A_d, A_sz*sizeof(float) );    
    cudaMalloc( (void**)&plan[_id].B_d, B_sz*sizeof(float) );
    cudaMalloc( (void**)&plan[_id].C_d, C_sz*sizeof(float) );
    cudaDeviceSynchronize();
}

/* 
    Periodic CPU works PRE kernel launching
    1. fresh some values
    2. memcpy, async on a specific stream
*/
void MMUL_TASK::taskPreGPU(cudaStream_t* s, unsigned int _id) {
    // WORKLOAD_INFO("MMUL", "Memory copying HtoD...");
    the_stream = s;
    cudaMemcpyAsync(plan[_id].A_d, A_h, A_sz*sizeof(float), cudaMemcpyHostToDevice, *the_stream);
    cudaMemcpyAsync(plan[_id].B_d, B_h, B_sz*sizeof(float), cudaMemcpyHostToDevice, *the_stream);
}   

/*  */
void MMUL_TASK::taskPostGPU(unsigned int _id) {
    // WORKLOAD_INFO("MMUL", "Memory copying DtoH...");
    cudaMemcpyAsync(plan[_id].C_h, plan[_id].C_d, C_sz*sizeof(float), cudaMemcpyDeviceToHost, *the_stream);
    
    // cudaStreamSynchronize(*the_stream);
    // verify_2(A_h, B_h, plan[_id].C_h, matArow, matBcol, matBrow);
}

/* 
    Launch kernel:
    1. SM allocation, decided by the algorithm
    2. launch the kernel(s)
*/
void MMUL_TASK::taskRunOnGPU(unsigned int _id, std::vector<int> &_sm_arr) {
    // int sm_num = _sm_arr.size();
    int __n = _sm_arr.size();
    for (int i = 0; i < plan[_id].sm_num - __n; i++) {
        _sm_arr.push_back(-1);
    }
    STGM_SM_MAPPING(plan[_id].sm_num, _sm_arr);
    // STGM_INIT_OLD(sm_num, _sm_arr);
    // WORKLOAD_INFO("MMUL", "Launching kernel...");
    STGM_LAUNCH_KERNEL(_id, mysgemm, *the_stream, DimGrid, DimBlock, matArow, matBcol, matBrow, plan[_id].A_d, plan[_id].B_d, plan[_id].C_d);

}

/*  */
void MMUL_TASK::taskFinish() {
    for (int _id = 0; _id < 2; _id++) {
        cudaFreeHost(plan[_id].C_h);
        cudaFree(plan[_id].A_d);
        cudaFree(plan[_id].B_d);
        cudaFree(plan[_id].C_d);
        STGM_FINISH(_id);
    }
    free(A_h);
    free(B_h);
}