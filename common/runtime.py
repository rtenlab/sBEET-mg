from dataclasses import dataclass
from typing import List
import pandas as pd
from functools import cmp_to_key

from src.model.task_model import Task
from src.model.job_model import Job
from src.model.gpu_model import GPU
from common.define import *
from src.model.support import sys_time

@dataclass
class running_task_t:
	task_id: int
	gpu_id: int
	slot_id: int
	est_finish_time: float
	abs_deadline: float

task_list: List[Task] = []
gpu_list: List[GPU] = []

class JobManager():
	def __init__(self, _verbose: bool, _emode: bool):
		self.released: int = 0
		self.missed: int = 0
		self.completed: int = 0
		self.ready_queue: List[Job] = []
		self.job_latest: List[Job] = []
		self.running_tasks: List[running_task_t] = []
		self.energy_pred = 0.0
		self.verbose = _verbose
		self.emode = _emode
		self.emode_recorder = [[], []]

	def sort_ready_queue(self, _order: str):
		if _order == "RM":
			self.ready_queue.sort(key=cmp_to_key(job_period_cmp))
		elif _order == "FIFO":
			self.ready_queue.sort(key=cmp_to_key(job_release_cmp))

	def release(self) -> None:
		period_prev = 0.0
		for i in range(len(task_list)):
			if self.emode == True and self.emode_recorder[0][i] == self.emode_recorder[1][i]: # target number reached, not push to ready_queue
				continue

			if self.job_latest[i].task_id == -1: # first job
				period_prev = task_list[i].phi
			else:
				period_prev = self.job_latest[i].release_time + task_list[self.job_latest[i].task_id].period
			
			if sys_time.get_sys_time() >= period_prev:
				self.job_latest[i] = Job(task_list[i].task_id, task_list[i].app_id, task_list[i].gpu_id, sys_time.get_sys_time(), sys_time.get_sys_time() + task_list[i].period * deadline_factor)
				self.ready_queue.append(self.job_latest[i])
				self.released += 1
				printWrapper(self.verbose, "[Task {}] is released".format(task_list[i].task_id))
		self.sort_ready_queue("RM")

	def complete(self) -> None: # by checking running_tasks
		rev_index = []
		for i, rt in enumerate(self.running_tasks):
			if rt.est_finish_time <= sys_time.get_sys_time():
				if rt.abs_deadline >= sys_time.get_sys_time():
					if self.emode == True:
						self.emode_recorder[1][rt.task_id] += 1
						# print("Task {} completed {}".format(rt.task_id, self.emode_recorder[1][rt.task_id]))
					self.completed += 1
					printWrapper(self.verbose, "[Task {}] is completed".format(rt.task_id))
				else:
					self.missed += 1
					printWrapper(self.verbose, "[Task {}] is missed".format(rt.task_id))
				gpu_list[rt.gpu_id].reset_slot(rt.slot_id)
				rev_index.append(i)
		for ri in sorted(rev_index, reverse=True):
			del self.running_tasks[ri]

	def execute(self, job: Job) -> None:
		job.update_config(job.gpu_id, job.get_sm(), sys_time.get_sys_time())
		sid = gpu_list[job.gpu_id].assign_slot(job, -1)
		gpu_list[job.gpu_id].reset_slot_waitlist(sid)
		self.running_tasks.append(running_task_t(job.task_id, job.gpu_id, sid, job.end_time, job.deadline))
		printWrapper(self.verbose, "[Task {}] starts execution with SM {}({})".format(job.task_id, job.sm, job.gpu_id))

	def skip(self) -> None: # original abort()
		rev_index = []
		for i in range(len(self.ready_queue)):
			if sys_time.get_sys_time() >= self.ready_queue[i].deadline: 
				rev_index.append(i)
				self.missed += 1
		for ri in sorted(rev_index, reverse=True):
			del self.ready_queue[ri]

''' Some methods '''
def find_gpu_with_min_cap(task: Task) -> int:
	temp_util = []
	for gpu in gpu_list:
		temp_util.append(gpu.util + task.get_util(gpu.gpu_id))
	index = temp_util.index(min(temp_util))
	return gpu_list[index].gpu_id

def job_period_cmp(a: Job, b: Job) -> int:
	if task_list[a.task_id].period < task_list[b.task_id].period:
		return -1
	else:
		return 1

def job_release_cmp(a: Job, b: Job) -> int:
	if a.release_time < b.release_time:
		return -1
	else:
		return 1

def printTaskset(verbose, policy_name, _task_list):
	if policy_name == "mg-jm":
		for ts in _task_list:
			u = ts.get_util(ts.gpu_id)
			df_tmp = {
				"Task ID": ts.task_id,
				"APP ID": ts.app_id,
				"Period": ts.period,
				"Phi": ts.phi,
				"GPU": ts.gpu_id,
				"Mopt": ts.mopt,
				"Util": u
			}
			printWrapper(verbose, df_tmp)
	else:
		for ts in _task_list:
			df_tmp = {
				"Task ID": ts.task_id,
				"APP ID": ts.app_id,
				"Period": ts.period,
				"Phi": ts.phi,
				"GPU": ts.gpu_id,
			}
			printWrapper(verbose, df_tmp)	

def printWrapper(_vb: bool, msg):
	if _vb is True:
		print("{} {}".format(sys_time.convert_to_string(), msg))
	else:
		pass