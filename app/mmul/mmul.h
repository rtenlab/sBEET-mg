#ifndef __MMUL_H__
#define __MMUL_H__

#include <cuda_runtime.h>
#include <vector>
#include "app/base_task.h"
#include "common/include/STGM.h"

typedef struct {
    float *A_d;
    float *B_d;
    float *C_d;
    float *C_h;
    STGM_DEFINE_VARS();
}mmul_plan;

/* 
    For each task instance, we assume the matrix size doesn't change
*/
class MMUL_TASK : public BASE_TASK {
    public:
        MMUL_TASK(int, int, int);
        ~MMUL_TASK();
        /* functions */
        // init, run once at the system power on
        void taskInit();
        void taskInitDevice(unsigned int);
        // finish, run once at the system power off
        void taskFinish();
        // TODO: probably need to extract memcpy to another function if other works on the cpu is too much
        void taskPreGPU(cudaStream_t*, unsigned int); 
        void taskPostGPU(unsigned int); 
        void taskRunOnGPU(unsigned int, std::vector<int> &);

        // void verify();

        /* variables  */
        cudaStream_t* the_stream;

        float *A_h;
        float *B_h;
        mmul_plan plan[2];
        size_t A_sz, B_sz, C_sz;
        unsigned matArow, matAcol, matBrow, matBcol;
        dim3 DimGrid, DimBlock;
        const unsigned int BLOCK_SIZE = 16;
};

#endif /* __MMUL_H__ */
