#ifndef __STEREODISPARITY_H__
#define __STEREODISPARITY_H__

#include <cuda_runtime.h>

#include "app/base_task.h"
#include "common/include/STGM.h"

 // result on the device side
typedef struct {
    unsigned int *h_odata;
    unsigned int *d_odata;
    unsigned int *d_img0;
    unsigned int *d_img1;
    STGM_DEFINE_VARS();
}stereo_plan;

/* 
    For each task instance, we assume the matrix size doesn't change
*/
class STEREODISPARITY_TASK : public BASE_TASK {
    public:
        STEREODISPARITY_TASK();
        ~STEREODISPARITY_TASK();
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

        /* variables  */
        cudaStream_t* the_stream;
        dim3 DimGrid, DimBlock;

        int minDisp, maxDisp;
        unsigned int w, h;
        unsigned int numData, memSize;
        unsigned int checkSum;
        unsigned char *h_img0;
        unsigned char *h_img1;
        stereo_plan plan[2];

        // cudaChannelFormatDesc ca_desc0;
        // cudaChannelFormatDesc ca_desc1;

    private:
        int iDivUp(int a, int b);
        
};

#endif /* __STEREODISPARITY_H__ */
