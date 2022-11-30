#ifndef __MMUL_KERNEL_H__
#define __MMUL_KERNEL_H__

#include <stdio.h>
#include <math.h>
#include "common/include/STGM.h"

// __device__ inline uint64_t GlobalTimer64(void) {
// 	volatile uint64_t reading;
// 	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(reading));
// 	return reading;
//  }
 
//  static __device__ __inline__ unsigned int GetSMID(void) {
// 	unsigned int ret;
// 	asm volatile("mov.u32 %0, %%smid;" : "=r"(ret));
// 	return ret;
//  }
STGM_DEFINE_KERNEL(mysgemm, int m, int n, int k, const float *A, const float *B, float* C) 
{
    int Row, Col;

	KERNEL_PROLOGUE();
	KERNEL_PROLOGUE_2();

    Row = BLOCKIDX_Y*blockDim.y+threadIdx.y;
    Col = BLOCKIDX_X*blockDim.x+threadIdx.x;

    if ((Row < m) && (Col < n)){
		float Cvalue = 0;
		for (int i = 0; i < k; ++i){
			Cvalue +=A[Row*k+i]*B[i*n+Col];
		}
		C[Row*n+Col] = Cvalue;	
    }
	KERNEL_EPILOGUE();
}

// STGM_DEFINE_KERNEL(mysgemm, int m, int n, int k, const float *A, const float *B, float* C, uint64_t *block_time, uint32_t *block_smid) 
// {
//     int Row, Col;

// 	KERNEL_PROLOGUE();


// 	uint64_t start_time = GlobalTimer64(); \
// 	if (threadIdx.x == 0) {  \
// 		block_time[blockIdx.x * 2] = start_time;  \
//         block_smid[blockIdx.x] = __get_smid();  \
// 	}  \
// 	__syncthreads();  \

// 	KERNEL_PROLOGUE_2()

//     Row = BLOCKIDX_Y*blockDim.y+threadIdx.y;
//     Col = BLOCKIDX_X*blockDim.x+threadIdx.x;

//     if ((Row < m) && (Col < n)){
// 		float Cvalue = 0;
// 		for (int i = 0; i < k; ++i){
// 			Cvalue +=A[Row*k+i]*B[i*n+Col];
// 		}
// 		C[Row*n+Col] = Cvalue;	
// 	}

// 	KERNEL_EPILOGUE();

// 	if (threadIdx.x == 0) {  \
//         block_time[blockIdx.x * 2 + 1] = GlobalTimer64();   \
//     }\
	
// }
#define TILE_SIZE 16
__global__ void mysgemm2(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    __shared__ float ds_A[TILE_SIZE][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x; 
    int ty = threadIdx.y;

    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;
    float Pvalue = 0; 

    int width_A = k; int height_A = m;
    int width_B = n; int height_B = k;
    int width_C = n; int height_C = m;

    // for (int p = 0; p < (width_C - 1) / TILE_SIZE + 1; ++p) {
    for (int p = 0; p < (width_A + TILE_SIZE - 1) / TILE_SIZE; ++p) {
        if (Row < height_A && p * TILE_SIZE + tx < width_A)
            ds_A[ty][tx] = A[Row * width_A + p * TILE_SIZE + tx];
        else
            ds_A[ty][tx] = 0.0;
        if (p * TILE_SIZE + ty < height_B && Col < width_B)
            ds_B[ty][tx] = B[(p * TILE_SIZE + ty) * width_B + Col];
        else
            ds_B[ty][tx] = 0.0;
        
        __syncthreads();

        if (Row < height_A && Col < width_B) {
            for (int i = 0; i < TILE_SIZE; ++i) {
                Pvalue += ds_A[ty][i] * ds_B[i][tx];
            }
        }
        __syncthreads();
    }
    if (Row < height_C && Col < width_C) 
        C[Row * width_C + Col] = Pvalue;
}

#endif // __MMUL_KERNEL_H__