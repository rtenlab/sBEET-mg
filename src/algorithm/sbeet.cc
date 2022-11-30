/* 
Reconstructed code of sBEET, Alg 2 and 3
 */
#include <bits/stdc++.h>
#include "common/include/defines.h"
#include "common/include/messages.h"
#include "src/model/gpu_slot_model.h"
#include "src/model/power_model.h"
#include "src/model/gpu_model.h"
#include "src/model/support.h"
#include "sbeet.h"

using namespace timing;

/* Returns a list of job that will arrive in the `time_window` */
std::vector<Job> get_job_list(std::vector<Task>& task_list, float *tw, unsigned int gpu_target, unsigned int task_id_excluded) {
	std::vector<Job> ret;
	for (auto & task : task_list) {
		if (task.task_id == task_id_excluded) {
			continue;
		}
		if (task.gpu_id != gpu_target) {
			continue;
		}
		unsigned int i = 0;
		while (task.phi + task.period * i + get_ghd(task.app_id, task.gpu_id) < tw[0]) {
			i++;
		}
		while (task.phi + task.period * i + get_ghd(task.app_id, task.gpu_id) >= tw[0] && task.phi + task.period * i + get_ghd(task.app_id, task.gpu_id) < tw[1]) {
			Job job(&task, task.gpu_id, (task.phi + task.period * i), (task.phi + task.period * i + task.period * deadline_factor));
			ret.push_back(job);
			i++;
		}
	}
	return ret;
}

const bool job_release_time_comparator (const Job &a, const Job &b) {
	return a.release_time < b.release_time;
}

/* 
Return the struct `SchedConfig`
The input argument `job` is already been configured, so no need to `update_config()` for it here
 */
SchedConfig schedule_generation(std::vector<GPU> & gpu_list, GPU *gpu, Job* job, std::vector<Job> & qw, float *tw) {
	gpu_status_t gs = gpu->get_status();
	SchedConfig config = {nullptr, std::numeric_limits<float>::max(), true};

	// if gpu is idle, and job is assigned with max_sm
	if (gs == IDLE && job->get_sm() == gpu->total_sm) {
		// directly check blocking for jobs in qw
		// std::cout << "checking blocking for task " << job->task->task_id << std::endl;
		float t_next = job->gpu_end_time;
		for (auto &jk : qw) {
			jk.update_config(gpu->gpu_id, std::pow(2, gpu->total_sm) - 1, t_next, 0);
			if (!jk.check_deadline()) {
				config.feasible = false;
				break;
			}
			t_next = jk.gpu_end_time;
		}
		config.energy = Power::predict_system_energy(gpu_list, tw);
		config.job = new Job(job->task, gpu->gpu_id, job->release_time, job->deadline);
		*config.job = *job;
		return config;
	}
	
	float t_next = 0;
	unsigned int sid_next = -1; // record the slot id of the next job in the waitlist that should be placed in
	unsigned int sid_temp = gpu->assign_slot(job, -1); // tentatively place
	if (gs ==  IDLE) {
		if (gpu->get_rem_sm() == 0) {
			t_next = tw[1];
			sid_next = sid_temp;
		}else {
			t_next = tw[0];
			sid_next = 1 - sid_temp;	
		}
	}else {
		int m_p = gpu->get_rem_sm();
		// if (gpu->slots[0].job->gpu_end_time <= gpu->slots[1].job->gpu_end_time) {
		// 	t_next = gpu->slots[0].job->gpu_end_time;
		// 	sid_next = 0;
		// }else {
		// 	t_next = gpu->slots[1].job->gpu_end_time;
		// 	sid_next = 1;
		// }
		float t_curr = 0;
		for (int i = 0; i < partition_num; i++) {
			if (!gpu->slots[i].empty_()) {
				t_curr = gpu->slots[i].job->gpu_end_time;
				sid_next = i;
				break;
			}
		}
		if (job->gpu_end_time < t_curr) {
			t_next = job->gpu_end_time;
			sid_next = 1 - sid_next;
		}else {
			t_next = t_curr;
		}
	}
	if (!job->check_deadline())
		config.feasible = false;
	// consider the jobs in q_w, sort q_w in job arriving time in ascending order
	std::sort(qw.begin(), qw.end(), job_release_time_comparator);
	unsigned int sm_rem = gpu->slots[sid_next].get_sm_tp(t_next); 
	// unsigned int sm_rem = gpu->total_sm - gpu->get_act_sm_tp(t_next);
	// std::cout << "sm_rem at " << t_next << " = " << sm_rem << std::endl;
	for (auto & jk : qw) {
		unsigned int m = MIN(sm_rem, jk.task->get_mopt(gpu->gpu_id));
		if (m == 0) {
			continue; // ignore this job for parallel execution
		}
		/* BEGINNING of sBEET Alg. 2 line 17 - 18 */
		float tp = MAX(t_next - get_ghd(jk.app_id, jk.task->gpu_id), jk.release_time);
		// tentatively assign, so randomly generate a sm_map
		jk.update_config(gpu->gpu_id, std::pow(2, m) - 1, tp, 0);
		// std::cout << "expected finished time of jk = " << jk.end_time << std::endl;
		gpu->slots[sid_next].push_to_waitlist(jk);
		// if (!gpu->slots[sid_next].waitlist.empty() && gpu->slots[1-sid_next].waitlist.empty()) { // when jobs should be append to the tail
		// 	t_next = gpu->slots[sid_next].waitlist.back().gpu_end_time;
		// }else 
		if (gpu->slots[1-sid_next].waitlist.empty() && !gpu->slots[1-sid_next].empty_()) {
			if (gpu->slots[sid_next].waitlist.back().gpu_end_time < gpu->slots[1-sid_next].job->gpu_end_time) {
				t_next = gpu->slots[sid_next].waitlist.back().gpu_end_time;
			}else {
				t_next = gpu->slots[1-sid_next].job->gpu_end_time;
				sid_next = 1 - sid_next;
			}
		}else { // Neither of the waitlist is empty, compare the finish time of the jobs in the waitlists
			if (gpu->slots[0].waitlist.back().gpu_end_time < gpu->slots[1].waitlist.back().gpu_end_time) {
				sid_next = 0;
				t_next = gpu->slots[0].waitlist.back().gpu_end_time;
			}else {
				sid_next = 1;
				t_next = gpu->slots[1].waitlist.back().gpu_end_time;
			}
		}
		/* END of sBEET Alg. 2 line 17 - 18 */
		if (!jk.check_deadline()) config.feasible = false;
		if (t_next > tw[1]) break;
	}
	config.energy = Power::predict_system_energy(gpu_list, tw);
	config.job = new Job(job->task, job->gpu_id, job->release_time, job->deadline);
	*config.job = *job;
	gpu->reset_slot(sid_temp);
	return config;
}

/* 
sBEET 
* It returns the config of a job
* sched_ok: whether the schedule of the returned job is feasible
*/
Job* sBeet(std::vector<Task>& task_list, std::vector<GPU> & gpu_list, GPU *gpu, Job *job, bool* sched_ok) {
	if (gpu->get_status() == IDLE && gpu->name == "t400") {
		job->update_config(gpu->gpu_id, gpu->gen_sm_map(gpu->total_sm), get_sys_time(), 0);
		if (job->check_deadline()) // rm
			*sched_ok = true;
		return job;
	}
	if (gpu->get_status() == IDLE && gpu->name == "rtx3070" && gpu->total_sm == 6) {
		job->update_config(gpu->gpu_id, gpu->gen_sm_map(gpu->total_sm), get_sys_time(), 0);
		if (job->check_deadline())
			*sched_ok = true;
		return job;
	}
	if (gpu->get_status() == IDLE) {
		std::vector<SchedConfig> sched_configs;
		bool exists = false; // indicates whether there is a feasible schedule
		for (unsigned int m = gpu->total_sm; m > 0; m-=4) {
		// for (unsigned int m = job->task->mopt; m > 0; m--) {
			job->update_config(gpu->gpu_id, gpu->gen_sm_map(m), get_sys_time(), 0);
			float tw[2] = {get_sys_time(), job->gpu_end_time};
			// std::cout << "time window: " << tw[0] << "," << tw[1] << std::endl;
			std::vector<Job> qw = get_job_list(task_list, tw, gpu->gpu_id, job->task->task_id);
			sched_configs.push_back(schedule_generation(gpu_list, gpu, job, qw, tw));
			if (sched_configs.back().feasible == true) {
				// std::cout << "feasible schedule when m=" << m << std::endl;
				exists = true;
			}
		}
		Job *ret = new Job(job->task, 0, 0, 0);
		float energy = std::numeric_limits<float>::max();
		/*  Choose the schedule with minimum energy, IF:
		 (1) There is at least one feasible schedule, choose one with the minimum energy; 
		(2) None of the generated schedule is feasible */
		for (auto & s : sched_configs) {
			if (s.feasible == true || exists == false) {
				// std::cout << "s.energy = " << s.energy << " ,energy = " << energy << std::endl;
				if (s.energy < energy) {
					energy = s.energy;
					*ret = *(s.job);
				}
			}
		}
		*sched_ok = exists;
		return ret;
	}else if (gpu->get_status() == ACTIVE) {
		unsigned int sid_act = gpu->get_active_slot();
		unsigned int idx = job->task->search_in_gpu_rankings(gpu->gpu_id);
		unsigned int m = gpu->get_rem_sm();
		job->update_config(gpu->gpu_id, gpu->gen_sm_map(m), get_sys_time(), 0);
		if (job->check_deadline()) {
			if (job->gpu_end_time <= gpu->slots[sid_act].job->end_time + 5) {
				*sched_ok = true;
				return job;
			}else {
				*sched_ok = false;
				return nullptr;
			}
		}else {
			float tw[2] = {get_sys_time(), job->gpu_end_time};
			std::vector<Job> qw = get_job_list(task_list, tw, gpu->gpu_id, job->task_id);
			SchedConfig config = schedule_generation(gpu_list, gpu, job, qw, tw);
			if (config.feasible == false) {
				*sched_ok = false;
				return nullptr;
			}else {
				Job * ret = new Job((config.job)->task, 0, 0, 0);
				*ret = *(config.job);
				*sched_ok = true;
				return ret;
			}
		}
	}else {
		return nullptr;
	}
}