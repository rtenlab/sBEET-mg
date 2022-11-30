#ifndef __STGM_H__
#define __STGM_H__

#include <cuda_runtime.h>
#include <vector>

#define STGM 

#ifndef PAR
#define PAR	8 // 8
#endif

#define NUMARGS(...)  (sizeof((int[]){__VA_ARGS__})/sizeof(int))

#ifdef STGM

// #define STGM_INIT(SM_NUM, ...)	\
// 	/* code snippet to transform into sm centric */	\
// 	int *mapping_d;	\
// 	int mapping_h[SM_NUM+1] = {SM_NUM, __VA_ARGS__}; /* 0: enable, -1: disable */	\
// 	if (NUMARGS(__VA_ARGS__) != SM_NUM) {	\
// 		printf("STGM_INIT: wrong arguments, ("#__VA_ARGS__")\n"); \
// 		exit(-1);\
// 	} \
// 	int active_sm = 0;	\
// 	for (int i = 1; i <= SM_NUM; i++)	\
// 		if (mapping_h[i] >= 0) active_sm++;	\
// 	mapping_h[0] = active_sm;	\
// 	/* memory allocation for mapping array */ \
// 	int mapping_size = (SM_NUM + 1) * sizeof(int);	\
// 	cudaMalloc((void**)&mapping_d, mapping_size);

#define STGM_INIT_OLD(SM_NUM, SM_VEC)	\
	/* code snippet to transform into sm centric */	\
	int *mapping_d;	\
	int mapping_h[SM_NUM+1]; \
    mapping_h[0] = SM_NUM; \
   for (int i = 0; i < SM_NUM; i++) { \
       mapping_h[i+1] = SM_VEC[i]; \
   } \
	if (SM_VEC.size() != SM_NUM) {	\
		printf("STGM_INIT: wrong arguments SM_VEC\n"); \
		exit(-1);\
	} \
	int active_sm = 0;	\
	for (int i = 1; i <= SM_NUM; i++)	\
		if (mapping_h[i] >= 0) active_sm++;	\
	mapping_h[0] = active_sm;	\
	/* memory allocation for mapping array */ \
	int mapping_size = (SM_NUM + 1) * sizeof(int);	\
	cudaMalloc((void**)&mapping_d, mapping_size);

// STGM_DEFINE_VARS(): Define variables required by STGM
// - Put this in a global scope in C, or in a class in C++
#define STGM_DEFINE_VARS() \
   int *mapping_d; \
   int mapping_size; \
   int sm_num;

#define STGM_INIT(DEV_ID) \
	cudaDeviceProp dev_prop; \
	cudaGetDeviceProperties(&dev_prop, DEV_ID); \
	plan[DEV_ID].sm_num = dev_prop.multiProcessorCount; \
	plan[DEV_ID].mapping_size = (plan[DEV_ID].sm_num + 1) * sizeof(int);	\
   cudaMalloc((void**)&plan[DEV_ID].mapping_d, plan[DEV_ID].mapping_size); 

#define STGM_FINISH(DEV_ID) \
  cudaFree(plan[DEV_ID].mapping_d);

#define STGM_SM_MAPPING(SM_NUM, SM_VEC)	\
   int mapping_h[SM_NUM+1]; \
   mapping_h[0] = SM_NUM; \
   for (int i = 0; i < SM_NUM; i++) { \
      mapping_h[i+1] = SM_VEC[i]; \
   } \
   if (SM_VEC.size() != SM_NUM) {	\
      printf("STGM_INIT: wrong arguments SM_VEC\n"); \
      exit(-1);\
   } \
   int active_sm = 0;	\
   for (int i = 1; i <= SM_NUM; i++)	\
      if (mapping_h[i] >= 0) active_sm++;	\
   mapping_h[0] = active_sm;	\

#define STGM_COPY_MAPPING(DEV_ID, STREAM)	\
	/* copy the mapping array to GPU memory	*/ \
	cudaMemcpyAsync(plan[DEV_ID].mapping_d, mapping_h, plan[DEV_ID].mapping_size, cudaMemcpyHostToDevice, STREAM); \

#define STGM_LAUNCH_KERNEL(DEV_ID, KERNEL, STREAM, GRID, THREADS, ...)	\
   STGM_COPY_MAPPING(DEV_ID, STREAM);	\
	/* printf("== THREAD_BLOCK_SIZE: %d, PAR: %d ==\n", dim3(THREADS).x * dim3(THREADS).y * dim3(THREADS).z, PAR);	\ */\
	/* kernel launch with mapping array */ \
	KERNEL<<< plan[DEV_ID].sm_num * PAR, THREADS, 0, STREAM >>>(plan[DEV_ID].mapping_d, GRID, ## __VA_ARGS__);

#define STGM_DEFINE_KERNEL(KERNEL, ...)	\
	__global__ void KERNEL(int *mapping, dim3 dimgrid, ## __VA_ARGS__)

#define __get_smid() ({ \
  uint ret;	\
  asm("mov.u32 %0, %smid;" : "=r"(ret) );	\
  ret; })

#define KERNEL_PROLOGUE()	\
   int smid = __get_smid();	\
   if (mapping[smid + 1] < 0) return;	\
   while (1) {	\
   __shared__ int worker_id;	\
   if (threadIdx.y == 0 && threadIdx.x == 0) {            \
      worker_id = 0; /* initialize	*/	\
      while (1) {	\
         int val = mapping[smid + 1];	\
         if (atomicCAS(&mapping[smid + 1], val, val + 1) == val) {	\
            worker_id = val;	\
            break;	\
         }	\
      }	\
   }	\
   __syncthreads();	\
   if (worker_id >= PAR) break;	\
   int active_sm = mapping[0];	\
   int sm_rel_id = 0;	\
   for (int i = 1; i < smid + 1; i++)	\
      if (mapping[i] >= 0) sm_rel_id++;	\
   int __start_offset = sm_rel_id + worker_id * active_sm;	\
   int __max_offset = dimgrid.x * dimgrid.y;	\
   int __stride = active_sm * PAR;	\
   /* if (threadIdx.y == 0 && threadIdx.x == 0) printf("%d, %d, %d: worker %d, block %d, SM %d \n", start_offset, max_offset, stride, worker_id, blockIdx.x, get_smid());  */ \

#define KERNEL_PROLOGUE_2()  \
   for (int __j = __start_offset; __j < __max_offset; __j += __stride) {

#define KERNEL_EPILOGUE() }}

#define BLOCKIDX_Y	(__j/dimgrid.x)
#define BLOCKIDX_X	(__j%dimgrid.x)
#define GRIDDIM_Y	(dimgrid.y)
#define GRIDDIM_X	(dimgrid.x)

#else // STGM

#define STGM_INIT(SM_NUM, ...)	
#define STGM_COPY()

#define STGM_LAUNCH_KERNEL(SM_NUM, KERNEL, STREAM, GRID, THREADS, ...)	\
	KERNEL<<< GRID, THREADS, 0, STREAM >>>(__VA_ARGS__)	

#define STGM_DEFINE_KERNEL(KERNEL, ...)	\
	__global__ void KERNEL(__VA_ARGS__)

#define KERNEL_PROLOGUE()	
#define KERNEL_PROLOGUE_2()	
#define KERNEL_EPILOGUE()

#define BLOCKIDX_Y	(blockIdx.y)
#define BLOCKIDX_X	(blockIdx.x)
#define GRIDDIM_Y	(gridDim.y)
#define GRIDDIM_X	(gridDim.x)

#endif // STGM

#endif // __STGM_H__
