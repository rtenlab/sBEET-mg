from dataclasses import dataclass
from typing import List, TypeVar, Tuple

from common.define import *
from src.model.power_model import predict_system_energy
from src.model.task_model import Task, get_ghd, get_gdh
from src.model.job_model import Job
from src.model.gpu_model import GPU
from src.model.support import sys_time

@dataclass
class SchedConfig:
	job: Job
	energy: float
	feasible: bool


def get_job_list(task_list: List[Task], wait_queue: List[Job], tw: List[float], gpu_target: int, task_id_excluded: int) -> List[Job]:
	ret = []
	for task in task_list:
		if task.task_id == task_id_excluded:
			continue
		if task.gpu_id != gpu_target:
			continue
		i = 0
		while task.phi + task.period * i + get_ghd(task.app_id, task.gpu_id) < tw[0]:
			i += 1
		while task.phi + task.period * i + get_ghd(task.app_id, task.gpu_id) >= tw[0] and task.phi + task.period * i + get_ghd(task.app_id, task.gpu_id) < tw[1]:
			job = Job(task.task_id, task.app_id, task.gpu_id, (task.phi + task.period * i), (task.phi + task.period * i + task.period * deadline_factor))
			ret.append(job)
			i += 1
	
	for jw in wait_queue:
		if jw.task.task_id != task_id_excluded:
			ret.append(jw)

	ret.sort(key=lambda x: x.release_time)	
	return ret

def schedule_generation(task_list: List[Task], gpu_list: List[GPU], gpu: GPU, job: Job, qw: List[Job], tw: List[float]) -> SchedConfig:
	gs = gpu.get_status()
	config = SchedConfig(None, sys.float_info.max, True)

	# if the gpu is idle, place the jobs at the tails one by one according to sBEET
	if gs == gpu_status_t.IDLE and job.get_sm() == gpu.total_sm:
		t_next = job.gpu_end_time
		for jk in qw:
			jk.update_config(gpu.gpu_id, gpu.total_sm, t_next)
			if jk.check_deadline() is False:
				config.feasible = False
				break
			t_next = jk.gpu_end_time
		config.energy = predict_system_energy(gpu_list, tw)
		config.job = job.clone()
		return config
	
	t_next = 0
	sid_next = -1
	sid_temp = gpu.assign_slot(job, -1)
	if gs == gpu_status_t.IDLE:
		if gpu.get_rem_sm() == 0:
			t_next = tw[1]
			sid_next = sid_temp
		else:
			t_next = tw[0]
			sid_next = 1 - sid_temp
	else:
		m_p = gpu.get_rem_sm()
		t_curr = 0
		for i in range(partition_num):
			if gpu.slots[i].empty_() is False:
				t_curr = gpu.slots[i].job.gpu_end_time
				sid_next = i
				break
		if job.gpu_end_time < t_curr:
			t_next = job.gpu_end_time
			sid_next = 1 - sid_next
		else:
			t_next = t_curr
	
	if job.check_deadline() is False:
		config.feasible = False
	
	sm_rem = gpu.slots[sid_next].get_sm_tp(t_next)
	for jk in qw:
		m = min(sm_rem, task_list[jk.task_id].get_mopt(gpu.gpu_id))
		if m == 0:
			continue
		# BEGINNING of sBEET Alg. 2 line 17 - 18
		tp = max(t_next - get_ghd(jk.app_id, gpu.gpu_id), jk.release_time)
		jk.update_config(gpu.gpu_id, m, tp)
		gpu.slots[sid_next].push_to_waitlist(jk)
		if len(gpu.slots[1-sid_next].waitlist) == 0 and gpu.slots[1-sid_next].empty_() is False:
			if gpu.slots[sid_next].waitlist[-1].gpu_end_time < gpu.slots[1-sid_next].job.gpu_end_time:
				t_next = gpu.slots[sid_next].waitlist[-1].gpu_end_time
			else:
				t_next = gpu.slots[1-sid_next].job.gpu_end_time
				sid_next = 1 - sid_next
		else: # neither of the waitlist is empty, compare the finish time of jobs in waitlist
			if gpu.slots[0].waitlist[-1].gpu_end_time < gpu.slots[1].waitlist[-1].gpu_end_time:
				t_next = gpu.slots[0].waitlist[-1].gpu_end_time
				sid_next = 0
			else:
				t_next = gpu.slots[1].waitlist[-1].gpu_end_time
				sid_next = 1
		# END of sBEET Alg. 2 line 17 - 18
		if jk.check_deadline() is False:
			config.feasible = False
		if t_next > tw[1]:
			break
	
	config.energy = predict_system_energy(gpu_list, tw)
	config.job = job.clone()
	gpu.reset_slot(sid_temp)
	return config

'''
Returns: sched_ok, job
'''
def sBEET(task_list: List[Task], gpu_list: List[GPU], gpu: GPU, job: Job, wait_queue: List[Job]) -> Tuple[bool, Job]:
	sched_ok = False
	if gpu.get_status() == gpu_status_t.IDLE and (gpu.name == "t400" or gpu.name == "T400"):
		job.update_config(gpu.gpu_id, gpu.total_sm, sys_time.get_sys_time())
		if job.check_deadline():
			sched_ok = True
		return sched_ok, job
	if gpu.get_status() == gpu_status_t.IDLE and (gpu.name == "rtx3070" or gpu.name == "RTX3070") and gpu.total_sm == 6:
		job.update_config(gpu.gpu_id, gpu.total_sm, sys_time.get_sys_time())
		if job.check_deadline():
			sched_ok = True
		return sched_ok, job
	
	if gpu.get_status() == gpu_status_t.IDLE:
		configs: List[SchedConfig] = []
		exists = False
		for m in range(gpu.total_sm, 0, -4):
			job.update_config(gpu.gpu_id, m, sys_time.get_sys_time())
			tw = [sys_time.get_sys_time(), job.gpu_end_time]
			qw = get_job_list(task_list, wait_queue, tw, gpu.gpu_id, job.task_id)
			configs.append(schedule_generation(task_list, gpu_list, gpu, job, qw, tw))
			if configs[-1].feasible is True:
				exists = True
		
		ret = None
		energy: float = sys.float_info.max
		# Choose the schedule with the minimum energy 
		for cfg in configs:
			if cfg.feasible is True or exists is False:
				if cfg.energy < energy:
					energy = cfg.energy
					ret = cfg.job.clone()
		# todo: patch - run as fast as possible if not exists. It should be reduced to RM instead of using part of the GPU causing a longer blocking time
		return exists, ret
	elif gpu.get_status() == gpu_status_t.ACTIVE:
		sid_act = gpu.get_active_slot()
		m = gpu.get_rem_sm()
		job.update_config(gpu.gpu_id, m, sys_time.get_sys_time())
		if job.check_deadline():
			if job.gpu_end_time <= gpu.slots[sid_act].job.end_time + 5:
				return True, job
			else:
				return False, None
		else:
			tw = [sys_time.get_sys_time(), job.gpu_end_time]
			qw = get_job_list(task_list, wait_queue, tw, gpu.gpu_id, job.task_id)
			config = schedule_generation(task_list, gpu_list, gpu, job, qw, tw)
			if config.feasible is False:
				return False, None
			else:
				ret = config.job.clone()
				return True, ret
	else: # if gpu is full
		return None