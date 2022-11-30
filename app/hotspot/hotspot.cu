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

#include "hotspot_kernel.cuh"
#include "hotspot.h"
#include "common/include/STGM.h"
#include "common/include/messages.h"

/* 
    Take the input arguments for this class
    a1, a2, a3 > 0
*/
HOTSPOT_TASK::HOTSPOT_TASK(int a1, int a2, int a3, char* a4, char* a5, char* a6) {
    grid_rows = a1; grid_cols = a1;
    pyramid_height = a2;
    total_iterations = a3;
    
    tfile = a4;
    pfile = a5;
    ofile = a6;

    size = grid_rows * grid_cols;

    borderCols = (pyramid_height)*EXPAND_RATE/2;
    borderRows = (pyramid_height)*EXPAND_RATE/2;
    smallBlockCol = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    smallBlockRow = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
    blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);

    DimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    DimGrid = dim3(blockCols, blockRows);
};

/*  */
HOTSPOT_TASK::~HOTSPOT_TASK() {};

/* 
    Init: 
    1, Malloc (host, device)
    2. definition of variables if necessary
*/
void HOTSPOT_TASK::taskInit() {
    // WORKLOAD_INFO("HOTSPOT", "Allocating variables...");
    FilesavingTemp = (float *) malloc(size*sizeof(float));
    FilesavingPower = (float *) malloc(size*sizeof(float));

    // printf("pyramidHeight: %d\ngridSize: [%d, %d]\nborder:[%d, %d]\nblockGrid:[%d, %d]\ntargetBlock:[%d, %d]\n", pyramid_height, grid_cols, grid_rows, borderCols, borderRows, blockCols, blockRows, smallBlockCol, smallBlockRow);

    readinput(FilesavingTemp, grid_rows, grid_cols, tfile);
    readinput(FilesavingPower, grid_rows, grid_cols, pfile);
};

void HOTSPOT_TASK::taskInitDevice(unsigned int _id) {
    STGM_INIT(_id);
    cudaMallocHost((void**)&plan[_id].MatrixOut, size * sizeof(float));

    cudaMalloc((void**)&plan[_id].MatrixTemp[0], sizeof(float)*size);
    cudaMalloc((void**)&plan[_id].MatrixTemp[1], sizeof(float)*size);
    cudaMalloc((void**)&plan[_id].MatrixPower, sizeof(float)*size);

    cudaDeviceSynchronize();
}

/* 
    Periodic CPU works PRE kernel launching
    1. fresh some values
    2. memcpy, async on a specific stream
*/
void HOTSPOT_TASK::taskPreGPU(cudaStream_t* s, unsigned int _id) {
    // WORKLOAD_INFO("HOTSPOT", "Memory copying HtoD...");
    the_stream = s;
    cudaMemcpyAsync(plan[_id].MatrixTemp[0], FilesavingTemp, sizeof(float)*size, cudaMemcpyHostToDevice, *the_stream);
    cudaMemcpyAsync(plan[_id].MatrixPower, FilesavingPower, sizeof(float)*size, cudaMemcpyHostToDevice, *the_stream);
}

/*  */
void HOTSPOT_TASK::taskPostGPU(unsigned int _id) {
    // WORKLOAD_INFO("HOTSPOT", "Memory copying DtoH...");
    int ret = dst;
    cudaMemcpyAsync(plan[_id].MatrixOut, plan[_id].MatrixTemp[ret], sizeof(float)*size, cudaMemcpyDeviceToHost, *the_stream);
    // NOTE: write output costs too much time
    // writeoutput(MatrixOut,grid_rows, grid_cols, ofile);
}

/* 
    Launch kernel:
    1. SM allocation, decided by the algorithm
    2. launch the kernel(s)
*/
void HOTSPOT_TASK::taskRunOnGPU(unsigned int _id, std::vector<int> &_sm_arr) {
	// int sm_num = _sm_arr.size();
	int __n = _sm_arr.size();
    for (int i = 0; i < plan[_id].sm_num - __n; i++) {
        _sm_arr.push_back(-1);
    }
    STGM_SM_MAPPING(plan[_id].sm_num, _sm_arr);

    // WORKLOAD_INFO("HOTSPOT", "Launching kernel...");
    float grid_height = chip_height / grid_rows;
	float grid_width = chip_width / grid_cols;
	float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	float Rz = t_chip / (K_SI * grid_height * grid_width);
	float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope;
    float time_elapsed = 0.00001;
    src = 1, dst = 0;

    for (int t = 0; t < total_iterations; t += pyramid_height) {
        int temp = src; src = dst; dst = temp;
        STGM_LAUNCH_KERNEL(_id, calculate_temp, *the_stream, DimGrid, DimBlock, MIN(pyramid_height, total_iterations - t), plan[_id].MatrixPower, plan[_id].MatrixTemp[src], plan[_id].MatrixTemp[dst], grid_cols, grid_rows, borderCols, borderRows, Cap, Rx, Ry, Rz, step, time_elapsed);
    }
}

/*  */
void HOTSPOT_TASK::taskFinish() {
    for (int _id = 0; _id < 2; _id++) {
        cudaFree(plan[_id].MatrixPower);
        cudaFree(plan[_id].MatrixTemp[0]);
        cudaFree(plan[_id].MatrixTemp[1]);
        cudaFreeHost(plan[_id].MatrixOut);
        STGM_FINISH(_id);
    }
    free(FilesavingTemp);
    free(FilesavingPower);
}

/* Tool functions */

void HOTSPOT_TASK::fatal(char *s) {
	fprintf(stderr, "error: %s\n", s);
}

void HOTSPOT_TASK::readinput(float *vect, int grid_rows, int grid_cols, char *file) {
    int i,j;
    FILE *fp;
    char str[STR_SIZE];
    float val;

    if( (fp  = fopen(file, "r" )) ==0 )
        printf( "The file was not opened\n" );

    for (i=0; i <= grid_rows-1; i++) {
        for (j=0; j <= grid_cols-1; j++) {
            fgets(str, STR_SIZE, fp);
            if (feof(fp))
                fatal("not enough lines in file");
            if ((sscanf(str, "%f", &val) != 1))
                fatal("invalid file format");
            vect[i*grid_cols+j] = val;
        }
    }
    fclose(fp);	
}

void HOTSPOT_TASK::writeoutput(float *vect, int grid_rows, int grid_cols, char *file) {
	int i,j, index=0;
	FILE *fp;
	char str[STR_SIZE];
	if( (fp = fopen(file, "w" )) == 0 )
        printf( "The file was not opened\n" );

	for (i=0; i < grid_rows; i++) {
        for (j=0; j < grid_cols; j++) {
            sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j]);
            fputs(str,fp);
            index++;
        }
    }	
    fclose(fp);	
}
