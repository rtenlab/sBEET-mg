#ifndef __HIST_KERNEL_H__
#define __HIST_KERNEL_H__

#include "common/include/STGM.h"

#define BLOCK_SIZE 512

STGM_DEFINE_KERNEL(histogram_kernel, unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins)
{
	KERNEL_PROLOGUE();
    KERNEL_PROLOGUE_2();
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
    for (unsigned int j = tid; j < num_elements; j += stride) {
        // by default, the randomly generated value should be in range (0, 4095)
        int position = input[j];
        if (position >= 0 && position <= num_bins - 1) {
			atomicAdd(&(bins[position]), 1);
		}
	}

	KERNEL_EPILOGUE();
}

#endif