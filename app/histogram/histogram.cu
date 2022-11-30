#include "histogram.h"
#include "histogram_kernel.cuh"
#include "common/include/STGM.h"
#include "common/include/messages.h"

HIST_TASK::HIST_TASK(unsigned int n_elements, unsigned int n_bins) {
	num_elements = n_elements;
	num_bins = n_bins;
}

void HIST_TASK::taskInit() {
	in_h = (unsigned int*)malloc(num_elements * sizeof(unsigned int));
	// cudaMallocHost((void**)in_h, num_elements * sizeof(unsigned int));
	
	for (unsigned int i = 0; i < num_elements; i++) {
		in_h[i] = rand() % num_bins;
	}
}

void HIST_TASK::taskInitDevice(unsigned int _id) {
  STGM_INIT(_id);
	// plan[_id].bins_h = (unsigned int *)cudaMallocHost(num_bins * sizeof(unsigned int));
	cudaMallocHost((void**)&plan[_id].bins_h, num_bins * sizeof(unsigned int));

	cudaMalloc((void**)&plan[_id].in_d, num_elements * sizeof(unsigned int));
    cudaMalloc((void**)&plan[_id].bins_d, num_bins * sizeof(unsigned int));

	cudaDeviceSynchronize();
}

void HIST_TASK::taskPreGPU(cudaStream_t *s, unsigned int _id) {
	the_stream = s;
	cudaMemcpyAsync(plan[_id].in_d, in_h, num_elements * sizeof(unsigned int), cudaMemcpyHostToDevice, *the_stream);
	cudaMemsetAsync(plan[_id].bins_d, 0, num_bins * sizeof(unsigned int), *the_stream);
}

void HIST_TASK::taskPostGPU(unsigned int _id) {
	cudaMemcpyAsync(plan[_id].bins_h, plan[_id].bins_d, num_bins * sizeof(unsigned int), cudaMemcpyDeviceToHost, *the_stream);
	// cudaStreamSynchronize(*the_stream);
	// verify_hist(in_h, plan[_id].bins_h, num_elements, num_bins);
}

void HIST_TASK::taskRunOnGPU(unsigned int _id, std::vector<int> &_sm_arr) {
	// int sm_num = _sm_arr.size();
	int __n = _sm_arr.size();
    for (int i = 0; i < plan[_id].sm_num - __n; i++) {
        _sm_arr.push_back(-1);
    }
    STGM_SM_MAPPING(plan[_id].sm_num, _sm_arr);

	const unsigned int GRID_SIZE = (int)ceil((float(num_elements)) / BLOCK_SIZE);
	dim3 DimGrid = dim3(GRID_SIZE);
	dim3 DimBlock = dim3(BLOCK_SIZE);

	STGM_LAUNCH_KERNEL(_id, histogram_kernel, *the_stream, DimGrid, DimBlock, plan[_id].in_d, plan[_id].bins_d, num_elements, num_bins);
}

void HIST_TASK::taskFinish() {
	for (int _id = 0; _id < 2; _id++) {
		cudaFreeHost(plan[_id].bins_h);
		cudaFree(plan[_id].in_d);
		cudaFree(plan[_id].bins_d);
		STGM_FINISH(_id);
	}
	cudaFreeHost(in_h);
}
