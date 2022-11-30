#ifndef __HOTSPOT_H__
#define __HOTSPOT_H__

#include <cuda_runtime.h>

#include "app/base_task.h"
#include "common/include/STGM.h"

#define STR_SIZE 256
#define MAX_PD	(3.0e6) /* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define PRECISION	0.001   /* required precision in degrees	*/
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
#define FACTOR_CHIP	0.5 /* capacitance fitting factor	*/

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)

typedef struct {
    float *MatrixOut; 
    float *MatrixTemp[2];
    float *MatrixPower;
    STGM_DEFINE_VARS();
}hotspot_plan;

/* 
    For each task instance, we assume the matrix size doesn't change
*/
class HOTSPOT_TASK : public BASE_TASK {
    public:
        HOTSPOT_TASK(int, int, int, char*, char*, char*);
        ~HOTSPOT_TASK();
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

        /* chip parameters	*/
        float t_chip = 0.0005;
        float chip_height = 0.016;
        float chip_width = 0.016;
        float amb_temp = 80.0;  /* ambient temperature, assuming no package at all	*/
        int size;
        int grid_rows, grid_cols;
        int pyramid_height;
        int total_iterations;
        char *tfile, *pfile, *ofile;
        /* --------------- pyramid parameters --------------- */
        # define EXPAND_RATE 2
        int borderCols;
        int borderRows;
        int smallBlockCol;
        int smallBlockRow;
        int blockCols;
        int blockRows;
        hotspot_plan plan[2];
        float *FilesavingTemp;
        float *FilesavingPower;
        
        dim3 DimGrid, DimBlock;

        int src, dst;

    private:
        void fatal(char *s);
        void readinput(float *vect, int grid_rows, int grid_cols, char *file);
        void writeoutput(float *vect, int grid_rows, int grid_cols, char *file);
        


};

#endif /* __HOTSPOT_H__ */
