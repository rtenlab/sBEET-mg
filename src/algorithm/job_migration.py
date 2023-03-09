from typing import List

from src.model.task_model import Task
from src.model.job_model import Job
from src.model.gpu_model import GPU
from src.model.power_model import *
from src.algorithm.sbeet import sBEET, get_job_list, schedule_generation
from common.define import *
from src.model.support import sys_time

def job_migration(task_list: List[Task], gpu_list: List[GPU], job: Job, wait_queue: List[Job]) -> Job:
	gpu = gpu_list[task_list[job.task_id].gpu_id]
	if gpu.get_status() == gpu_status_t.IDLE:
		energy_vec: List[float] = []
		jconf_vec: List[Job] = []
		job.update_config(gpu.gpu_id, task_list[job.task_id].mopt, sys_time.get_sys_time())
		tw = [sys_time.get_sys_time(), job.gpu_end_time]
		qw = get_job_list(task_list, wait_queue, tw, gpu.gpu_id, job.task_id)
		config = schedule_generation(task_list, gpu_list, gpu, job, qw, tw)
		job1 = None
		if config.feasible is True:
			sid = gpu.assign_slot(job, -1)
			e1 = predict_system_energy(gpu_list, tw)
			gpu.reset_slot(sid)
			job1 = job.clone()
			jconf_vec.append(job1)
			energy_vec.append(e1)
		else:
			job.update_config(gpu.gpu_id, gpu.total_sm, sys_time.get_sys_time())
			tw[1] = job.gpu_end_time
			sid = gpu.assign_slot(job, -1)
			e1 = predict_system_energy(gpu_list, tw)
			gpu.reset_slot(sid)
			job1 = job.clone()
			jconf_vec.append(job1)
			energy_vec.append(e1)
		e2 = sys.float_info.max
		for grkg in task_list[job.task_id].gpu_rankings: 
			if grkg.gpu_id == task_list[job.task_id].gpu_id:
				continue
			gpu_next = gpu_list[grkg.gpu_id]
			if gpu_next.get_status() == gpu_status_t.IDLE or gpu_next.get_status() == gpu_status_t.FULL:
				continue
			else:
				sched_ok, job2 = sBEET(task_list, gpu_list, gpu_next, job, wait_queue)
				if job2 is not None and job2.check_deadline():
					sid = gpu_next.assign_slot(job2, -1)
					tw = [job2.start_time, job2.end_time]
					energy_vec.append(predict_system_energy(gpu_list, tw))
					jconf_vec.append(job2)
					gpu_next.reset_slot(sid)
		if len(jconf_vec) > 0 and len(energy_vec) > 0:
			index = energy_vec.index(min(energy_vec))
			return jconf_vec[index]
	elif gpu.get_status() == gpu_status_t.ACTIVE:
		return sBEET(task_list, gpu_list, gpu, job, wait_queue)[1]
	else: # gpu is full
		sid = gpu.get_faster_slot() 
		finish_time = gpu.slots[sid].job.end_time
		if gpu.total_sm - gpu.get_act_sm_tp(finish_time) < task_list[job.task_id].mopt: # this guarantees [1-sid] is valid
			finish_time = gpu.slots[1-sid].job.end_time # todo: check tp() functions, tp's tie breaker
		job.update_config(gpu.gpu_id, task_list[job.task_id].mopt, finish_time)
		if job.check_deadline():
			return None
		
		# try on other GPUs
		energy_vec: List[float] = []
		jconf_vec: List[Job] = []
		for gpu_ in gpu_list:
			if gpu_.gpu_id == gpu.gpu_id:
				continue
			if gpu_.get_status() == gpu_status_t.FULL:
				continue
			sched_ok = False
			sched_ok, jconf = sBEET(task_list, gpu_list, gpu_, job, wait_queue)
			if jconf is not None and sched_ok:
				sid = gpu_.assign_slot(jconf, -1)
				tw = [jconf.start_time, jconf.end_time]
				energy_vec.append(predict_system_energy(gpu_list, tw))
				jconf_vec.append(jconf)
				gpu_.reset_slot(sid)
		if len(jconf_vec) > 0 and len(energy_vec) > 0:
			index = energy_vec.index(min(energy_vec))
			return jconf_vec[index]
		else:
			return None