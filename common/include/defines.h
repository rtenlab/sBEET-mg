#pragma once
#ifndef __DEFINES_H__
#define __DEFINES_H__

#define _GNU_SOURCE

#include <string>
#include <vector>
#include <limits>

#define MIN(a, b) (a < b) ? a : b
#define MAX(a, b) (a > b) ? a : b

// number of SM partitions per GPU
const int partition_num = 2;
// NOTE: If changes gpu_num, also remember to change the initialization of gpu_list, initGPUs()
const int gpu_num = 2;
const int sm_limit[gpu_num] = {24,6}; // The number of SMs each GPU is going to use
const int reference_sm[gpu_num] = {24,6}; // constant, for reference utilization computation

const float deadline_factor = 0.5; // deadline = period * factor

namespace rt_task {
	enum status_t {READY, FREED, GPU_DONE};

	/* The shared struct that keeps the status of the tasks at runtime */
	struct rt_task_t {
		status_t status;
		unsigned int task_id;
		bool valid; // tracking whether a "completion" is valid. Whether this available struct is released from a completed job
		bool completed; // the job completed or sctually missed deadline
		bool shutdown;
	};

	extern std::vector<rt_task_t> rt_vec;

	void free_rt_status(unsigned int rid, bool completed);
}

#endif // __DEFINES_H__