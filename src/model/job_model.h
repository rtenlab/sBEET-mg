#ifndef __JOB_MODEL_H__
#define __JOB_MODEL_H__

#include <stdint.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "app/base_task.h"
#include "task_model.h"
#include "support.h"


class Job {
public:
	Job(Task* task, unsigned int gpu_id, float release_time, float deadline);
	~Job();

	/* variables */
	/* const fields */
	unsigned int task_id; // the index of the task stored in task_list
	app_t app_id;
	Task *task;
	float release_time;
	float deadline;
	/* changing fields */
	unsigned int gpu_id;
	unsigned int cpu_id;
	unsigned int wid; // the id of the worker the job assigned to
	/* changing timing fields, they are absolute */
	float start_time;
	float gpu_start_time;
	float gpu_end_time;
	float end_time;

	bool first_launch;

	/* functions */
	float get_gpu_wcet() const; // get the wcet on the assigned gpu
	void update_timing(const float tp);
	void update_config(unsigned int _gpu_id, uint64_t _sm_map, const float tp, bool _exec);
	const bool check_deadline() const; // check whether the job with the assignation can meet the deadline
	const uint64_t get_sm_map() const;
	const unsigned int get_sm() const;
	bool check_sm_map_ok(unsigned int gpu_id, uint64_t sm_map);

	/* runtime */
	BASE_TASK *workload;
	std::vector<int> sm_arr;
	void _jobExec();
	void testExec();

	cudaStream_t *stream;

protected:
	void update_sm_map(uint64_t _sm_map, int size, bool _exec);

private:
	unsigned int sm;
	uint64_t sm_map;
};

#endif // !__JOB_MODEL_H__