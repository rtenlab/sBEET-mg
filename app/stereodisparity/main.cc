/*
    The program is used to profile this task
 */

// #define _GNU_SOURCE

#include <stdio.h>
#include <sched.h>

#include "common/include/STGM.h"
#include <support.h>
#include "stereodisparity.h"


int main(int argc, char *argv[]) {

    cpu_set_t  mask;
    CPU_ZERO(&mask);
    CPU_SET(2, &mask);
    int res = sched_setaffinity(0, sizeof(mask), &mask);

    CREATE_STREAM(1);
    int sm_arr[8] = { 0, 0, 0, 0, 0, 0, 0, 0};

    STEREODISPARITY_TASK stereodisparity_obj(streams[0]);

    Timer mytimer;
    float init_tot_time = 0;
    float pre_tot_time = 0;
    float post_tot_time = 0;
    float finish_tot_time = 0;

    LOOP_BEGIN(1000);

    startTime(&mytimer);
    stereodisparity_obj.taskInit();
    stopTime(&mytimer);
    init_tot_time += elapsedTime(mytimer);

    startTime(&mytimer);
    stereodisparity_obj.taskPreGPU();
    stopTime(&mytimer);
    pre_tot_time += elapsedTime(mytimer);

    stereodisparity_obj.taskRunOnGPU(sm_arr);    
    cudaDeviceSynchronize(); // only used along with loop

    startTime(&mytimer);
    stereodisparity_obj.taskPostGPU();
    stopTime(&mytimer);
    post_tot_time += elapsedTime(mytimer);

    startTime(&mytimer);
    stereodisparity_obj.taskFinish();
    stopTime(&mytimer);
    finish_tot_time += elapsedTime(mytimer);

    LOOP_END();

    // unit: sec
    std::cout << init_tot_time * 1000 / 1000 << std::endl;
    std::cout << pre_tot_time * 1000 / 1000 << std::endl;
    std::cout << post_tot_time *1000 / 1000 << std::endl;
    std::cout << finish_tot_time * 1000 / 1000 << std::endl;

    return 0;
    
}