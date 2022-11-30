#ifndef __SBEET_H__
#define __SBEET_H__

#include <vector>

#include "src/model/job_model.h"
#include "src/model/gpu_model.h"
#include "src/model/task_model.h"

struct SchedConfig {
	Job *job;
	float energy;
	bool feasible;
};

std::vector<Job> get_job_list(std::vector<Task>& task_list, float *tw, unsigned int gpu_target, unsigned int task_id_excluded);

SchedConfig schedule_generation(std::vector<GPU> & gpu_list, GPU *gpu, Job* job, std::vector<Job> & qw, float *tw);

Job* sBeet(std::vector<Task>& task_list, std::vector<GPU> & gpu_list, GPU *gpu, Job *job, bool* sched_ok);


#endif // !__SBEET_H__