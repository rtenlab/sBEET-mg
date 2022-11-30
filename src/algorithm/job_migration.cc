/* 
Alg. 2, algorithm of runtime job migration
 */

#include <bitset>
#include <math.h>

#include "common/include/defines.h"
#include "common/include/messages.h"
#include "sbeet.h"
// #include "common/include/timing.h"
#include "src/model/job_model.h"
#include "src/model/power_model.h"
#include "src/model/support.h"

using namespace Power;
using namespace timing;

Job* sBeetWrapper(std::vector<Task> & task_list, std::vector<GPU> & gpu_list, GPU* gpu, Job *job) {
	bool sched_ok = false;
	return sBeet(task_list, gpu_list, gpu, job, &sched_ok);
}

/* 
 * note that this function should only HELP energy, and should not put energy over schedularity when making decisions
 */
Job* job_migration(std::vector<Task> & task_list, std::vector<GPU> & gpu_list, Job *job) {
	GPU *gpu = &gpu_list[job->task->gpu_id];
	if (gpu->get_status() == IDLE) {
		std::vector<float> energy_vec;
		std::vector<Job *> jconf_vec;
		/* If the job is assigned to the best GPU */
		job->update_config(gpu->gpu_id, gpu->gen_sm_map(job->task->mopt), get_sys_time(), 0);
		float tw[2] = {get_sys_time(), job->gpu_end_time};
		std::vector<Job> qw = get_job_list(task_list, tw, gpu->gpu_id, job->task->task_id);
		SchedConfig config = schedule_generation(gpu_list, gpu, job, qw, tw);
		Job *job1 = new Job(job->task, 0, 0, 0);
		if (config.feasible == true) {
			unsigned int sid = gpu->assign_slot(job, -1); 
			float e1 = predict_system_energy(gpu_list, tw);
			gpu->reset_slot(sid);
			*job1 = *job;
			jconf_vec.push_back(job1);
			energy_vec.push_back(e1);
		}else { // patch, what if running it as fast as possible
			job->update_config(gpu->gpu_id, gpu->gen_sm_map(gpu->total_sm), get_sys_time(), 0);
			{
			float tw[2] = {get_sys_time(), job->gpu_end_time};
			unsigned int sid = gpu->assign_slot(job, -1); 
			float e1 = predict_system_energy(gpu_list, tw);
			gpu->reset_slot(sid);
			*job1 = *job;
			jconf_vec.push_back(job1);
			energy_vec.push_back(e1);
			}
		}
		float e2 = std::numeric_limits<float>::max();
		for (auto & it : job->task->gpu_rankings) {
			if (it.gpu_id == job->task->gpu_id) {
				continue;
			}
			GPU *gpu_next = &gpu_list[it.gpu_id];
			if (gpu_next->get_status() == IDLE || (gpu_next->get_status() == FULL)) {
				continue;
			}else {
				bool sched_ok;
				Job * job2 = sBeet(task_list, gpu_list, gpu_next, job, &sched_ok);
				// compute the energy
				if (job2 != nullptr && job2->check_deadline()) {
					unsigned int sid = gpu_next->assign_slot(job2, -1);
					float tw[2];
					tw[0] = job2->start_time;
					tw[1] = job2->end_time;
					energy_vec.push_back(predict_system_energy(gpu_list, tw));
					jconf_vec.push_back(job2);
					gpu_next->reset_slot(sid);
				}
			}
		}
		if (!jconf_vec.empty() && !energy_vec.empty()) {
			int index = -1;
			float val = std::numeric_limits<float>::max();
			for (int j = 0; j < energy_vec.size(); j++) {
				if (energy_vec[j] < val) {
					val = energy_vec[j];
					index = j;
				}
			}
			return jconf_vec[index];
		}else {
			return nullptr;
		}
	}else if (gpu->get_status() == ACTIVE) {
		bool sched_ok = false;
		return sBeet(task_list, gpu_list, gpu, job, &sched_ok);
	}else { 
		// if the gpu if full, why not just try sBeet on other GPUs
		// however, before this, need to make sure the job will be feasible on the offline assigned GPU
		unsigned int sid = gpu->get_faster_slot();
		float finish_time = gpu->slots[sid].job->end_time;
		if (gpu->total_sm - gpu->get_act_sm_tp(finish_time) < job->task->mopt) {
			finish_time = gpu->slots[1-sid].job->end_time;
		}
		job->update_config(gpu->gpu_id, std::pow(2, job->task->mopt) - 1, finish_time, 0);
		if (job->check_deadline()) {
			// std::cout << "will wait for offline assigned gpu" << std::endl;
			return nullptr;
		}

		// not returned, try on other GPUs
		std::vector<float> energy_vec;
		std::vector<Job *> jconf_vec;
		for (int i = 0; i < gpu_list.size(); i++) {
			if (gpu_list[i].gpu_id == gpu->gpu_id) 
				continue;
			if (gpu_list[i].get_status() == FULL)
				continue;
			bool sched_ok = false;
			Job *jconf = sBeet(task_list, gpu_list, &gpu_list[i], job, &sched_ok);
			// find the most energy-efficient schedule
			if (jconf != nullptr && sched_ok == true) {
				unsigned int sid = gpu_list[i].assign_slot(jconf, -1);
				float tw[2];
				tw[0] = jconf->start_time;
				tw[1] = jconf->end_time;
				energy_vec.push_back(predict_system_energy(gpu_list, tw));
				jconf_vec.push_back(jconf);
				gpu_list[i].reset_slot(sid);
				// std::cout << "task " << jconf->task_id << " safe to be on GPU " << jconf->gpu_id << std::endl;
			}
		}
		if (!jconf_vec.empty() && !energy_vec.empty()) {
			int index = -1;
			float val = std::numeric_limits<float>::max();
			for (int j = 0; j < energy_vec.size(); j++) {
				if (energy_vec[j] < val) {
					val = energy_vec[j];
					index = j;
				}
			}
			return jconf_vec[index];
		}else {
			return nullptr;
		}
	}
}