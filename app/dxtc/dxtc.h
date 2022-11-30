#ifndef __DXTC_H__
#define __DXTC_H__

#include <cuda_runtime.h>
#include <string>

#include "app/base_task.h"
#include "common/include/STGM.h"

typedef struct {
    unsigned int *d_data;
    unsigned int *d_result;
    unsigned int *d_permutations;
    unsigned int *h_result;
    STGM_DEFINE_VARS();
}dxtc_plan;

/* 
    For each task instance, we assume the matrix size doesn't change
*/
class DXTC_TASK : public BASE_TASK {
    public:
        DXTC_TASK(std::string);
        ~DXTC_TASK();
        /* functions */
        // init, run once at the system power on
        void taskInit();
        void taskInitDevice(unsigned int);
        // finish, run once at the system power off
        void taskFinish();
        // TODO: probably need to extract memcpy to another function if other works on the cpu is too much
        void taskPreGPU(cudaStream_t*, unsigned int); 
        void taskPostGPU(unsigned int _id); 
        void taskRunOnGPU(unsigned int, std::vector<int> &);

        /* variables  */
        cudaStream_t* the_stream;
        // dim3 DimGrid, DimBlock;
        std::string input_image;
        char *reference_image_path;
        unsigned char *data;
        unsigned int w, h;
        unsigned int memSize, compressedSize;
        unsigned int permutations[1024];
        dxtc_plan plan[2];
        unsigned int *block_image;
};

#endif /* __DXTC_H__ */
