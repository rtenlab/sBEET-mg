#ifndef __HIST_H__
#define __HIST_H__

#include <vector>
#include "app/base_task.h"
#include "common/include/STGM.h"

typedef struct {
	unsigned int* bins_h; // store results
    unsigned int *in_d;
    unsigned int* bins_d;
	STGM_DEFINE_VARS();
}hist_plan;

class HIST_TASK : public BASE_TASK {
public:
	HIST_TASK(unsigned int n_elements, unsigned int n_bins);
	
	void taskInit();
	void taskInitDevice(unsigned int);
	void taskPreGPU(cudaStream_t *s, unsigned int);
	void taskPostGPU(unsigned int);
	void taskFinish();
	void taskRunOnGPU(unsigned int, std::vector<int> &);

	cudaStream_t *the_stream;
    
    unsigned int num_elements, num_bins;
	hist_plan plan[2];
	unsigned int *in_h;
};

#endif // !__HIST_H__
