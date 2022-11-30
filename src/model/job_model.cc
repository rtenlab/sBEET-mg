#include <sched.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdexcept>
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)

#include "job_model.h"
#include "support.h"
#include "common/include/defines.h"
#include "common/include/messages.h"


using namespace timing;

Job::Job(Task *task, unsigned int gpu_id, float release_time, float deadline) {
	this->task = task;
	this->task_id = task->task_id;
	this->app_id = task->app_id;
	this->release_time = release_time;
	this->deadline = deadline; 

	this->gpu_id = gpu_id;
	this->sm = 0;
	this->sm_map = 0x0;

	this->start_time = 0.0f;
	this->end_time = 0.0f;
	this->gpu_start_time = 0.0f;
	this->gpu_end_time = 0.0f;

	this->workload = task->workload;
	this->first_launch = false;
}

Job::~Job() {}

void Job::update_timing(const float tp)  {
	this->start_time = tp;
	this->gpu_start_time = this->start_time + get_ghd(this->app_id, this->gpu_id);
	this->gpu_end_time = this->gpu_start_time + this->get_gpu_wcet();
	this->end_time = this->gpu_end_time + get_gdh(this->app_id, this->gpu_id);
}

void Job::update_config(unsigned int _gpu_id, uint64_t _sm_map, const float tp, bool _exec)  {
	this->gpu_id = _gpu_id;
	update_sm_map(_sm_map, sm_limit[_gpu_id], _exec);
	update_timing(tp);
}

	// check whether the job with the assignation can meet the deadline. Return `true` for met
const bool Job::check_deadline() const  {
	return this->end_time <= this->deadline;
}

const uint64_t Job::get_sm_map() const {
	return this->sm_map;
}

const unsigned int Job::get_sm() const  {
	return this->sm;
}

void Job::update_sm_map(uint64_t _sm_map, int size, bool _exec) {
	this->sm_map = _sm_map;
	this->sm = get_sm_by_map(_sm_map);
	if (_exec) {
		this->sm_arr.clear();
		for (int i = 0; i < size; i++) {
			if (_sm_map & 1) this->sm_arr.push_back(0);
			else this->sm_arr.push_back(-1);
			_sm_map >>= 1;
		}
	}
}

float Job::get_gpu_wcet() const { // search for the corresponding gpu_id in the gpu_rankings in Task
	bool failed = true;
	for (int i = 0; i < this->task->gpu_rankings.size(); i++) {
		if  (this->gpu_id == this->task->gpu_rankings[i].gpu_id) {
			return this->task->gpu_rankings[i].wcets[this->sm - 1];
		}
	}
	if (failed) {
		throw std::runtime_error("float Job::get_gpu_wcet() const failed.");
	}
}

/* check whether the number of SMs in input sm map is smaller than the total number of SMs on the target GPU */
// YIDI: Assuming the `gpu_id` is in order
bool Job::check_sm_map_ok(unsigned int gpu_id, uint64_t sm_map)  {
	if (gpu_id == 0) { // rtx 3070
		return sm_map <= (1 << RTX3070::total_sm) - 1;
	}else if (gpu_id == 1) { // t400
		return sm_map <= (1 << T400::total_sm) - 1;
	}
}

using namespace rt_task;

void rt_task::free_rt_status(unsigned int rid, bool completed) {
	rt_vec[rid].status = FREED;
	rt_vec[rid].valid = true;
	rt_vec[rid].completed = completed;
}

std::vector<rt_task_t> rt_task::rt_vec;

// #define TEST_NO_WORKLOAD

void Job::_jobExec() {
	unsigned int rid = (this->cpu_id - 8) / 2; // see worker.h
	rt_vec[rid].task_id = this->task_id;
	// abort if exceeding time or if the system is shut down
	// if (get_sys_time() >= this->deadline || rt_vec[rid].shutdown) {
	if (rt_vec[rid].shutdown) {
		shared_mutex.lock();
		free_rt_status(rid, false);
		shared_mutex.unlock();
		return;
	}

#ifndef TEST_NO_WORKLOAD
	if (this->first_launch == true) {
		// PRINT_VAL("prelaunch setting gpu device ", gpu_id);
		cudaSetDevice(gpu_id);
	}
	// h2d
	// PRINT_MSG("Memcpy H2D");
	this->workload->taskPreGPU(this->stream, this->gpu_id);
	cudaStreamSynchronize(*stream);
	// {
	// 	cudaError_t err = cudaStreamSynchronize(*this->stream);
	// 	if (err != cudaSuccess) {
	// 		PRINT_MSG("Memcpy H2D failed");
	// 	}else {
	// 		PRINT_MSG("Memcpy H2D succeeded");
	// 	}
	// }
#endif
	// if (get_sys_time() >= this->deadline || rt_vec[rid].shutdown) {
	if (rt_vec[rid].shutdown) {
		shared_mutex.lock();
		free_rt_status(rid, false);
		shared_mutex.unlock();
		return;
	}

#ifndef TEST_NO_WORKLOAD
	// exec
	// PRINT_VAL("Kernel exec", gpu_id);
	this->workload->taskRunOnGPU(this->gpu_id, this->sm_arr);
	cudaStreamSynchronize(*stream);
	// {
	// 	cudaError_t err = cudaStreamSynchronize(*this->stream);
	// 	if (err != cudaSuccess) {
	// 		PRINT_MSG("Kernel launching failed ");
	// 	}else {
	// 		PRINT_MSG("Kernel launching succeeded");
	// 	}
	// }
#endif
	shared_mutex.lock();
	rt_vec[rid].status = GPU_DONE;
	shared_mutex.unlock();
#ifndef TEST_NO_WORKLOAD
	// d2h
	// PRINT_MSG("Memcpy D2H");
	this->workload->taskPostGPU(this->gpu_id);
	cudaStreamSynchronize(*stream);
	// {
	// 	cudaError_t err = cudaStreamSynchronize(*this->stream);
	// 	if (err != cudaSuccess) {
	// 		PRINT_MSG("Memcpy D2H failed");
	// 	}else {
	// 		PRINT_MSG("Memcpy D2H succeeded");
	// 	}
	// }
#endif

	if (get_sys_time() >= this->deadline || rt_vec[rid].shutdown) { // missed
		shared_mutex.lock();
		free_rt_status(rid, false);
		shared_mutex.unlock();
	}else { // not missed
		shared_mutex.lock();
		free_rt_status(rid, true);
		shared_mutex.unlock();
	}
	return;
}