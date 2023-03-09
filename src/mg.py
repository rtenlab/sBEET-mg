import csv
from typing import List, Set, Union

from common.runtime import *
from src.model.power_model import *
from src.algorithm.job_migration import job_migration

class mg:
	def __init__(self, filename: str, verbose: bool, duration: float, policy_name: str, emode: bool):
		self.filename = filename
		self.verbose = verbose
		self.duration = duration
		self.policy_name = policy_name
		self.emode = emode
		self.jobMgr = JobManager(self.verbose, self.emode)
		for i in range(len(task_list)):
			self.jobMgr.job_latest.append(Job(-1, -1, -1, -1, -1))

	def taskDistribution(self) -> None:
		task_list.sort(key=lambda x: x.period)
		for task in task_list:
			isAssigned = False
			for tg in task.gpu_rankings:
				if gpu_list[tg.gpu_id].util + task.get_util(tg.gpu_id) <= 1.0:
					task.gpu_id = tg.gpu_id
					task.mopt = tg.mopt
					gpu_list[tg.gpu_id].util += task.get_util(tg.gpu_id)
					isAssigned = True
					break
			if isAssigned is False:
				task.gpu_id = find_gpu_with_min_cap(task)
				idx = task.search_in_gpu_rankings(task.gpu_id)
				task.mopt = task.gpu_rankings[idx].mopt
				gpu_list[task.gpu_id].util += task.get_util_mopt(task.gpu_id)
				isAssigned = True
		task_list.sort(key=lambda x: x.period)
		for i in range(len(task_list)):
			task_list[i].task_id = i
			# task_list[i].phi = (len(task_list) - 1 - i) * 2 # todo: remove
		
		if self.emode == True:
			for i in range(len(task_list)):
				target_num = int(self.duration / 20 / task_list[i].period) + 1
				self.jobMgr.emode_recorder[0].append(target_num)
				self.jobMgr.emode_recorder[1].append(0)
				# print("duration {}, period {}, target_num {}".format(self.duration, task_list[i].period, target_num))

	def mgOfflineHelper(self, job: Job) -> Union[Job, None]:
		_gpu_id = task_list[job.task_id].gpu_id
		if gpu_list[_gpu_id].get_status() is not gpu_status_t.FULL and gpu_list[_gpu_id].get_rem_sm() >= task_list[job.task_id].mopt:
			job.update_config(_gpu_id, task_list[job.task_id].mopt, sys_time.get_sys_time())
			return job
		else:
			return None

	def schedulerOnline(self) -> None:
		sys_time.init_sys_time()
		while sys_time.get_sys_time() < self.duration:
			if self.emode == 1:
				_n = 0;
				for i in range(len(self.jobMgr.emode_recorder[0])):
					if self.jobMgr.emode_recorder[0][i] == self.jobMgr.emode_recorder[1][i]:
						_n += 1
					# else:
					# 	print("{} is not completed. ".format(i))
				# print("_n = {}, {}".format(_n, len(self.jobMgr.emode_recorder[0])))
				if _n == len(self.jobMgr.emode_recorder[0]):
					# print("_n = {}".format(_n))
					break

			self.jobMgr.complete()
			self.jobMgr.skip()
			self.jobMgr.release()

			rev_index = []
			for i in range(len(self.jobMgr.ready_queue)):
				isRunning = False
				for rt in self.jobMgr.running_tasks:
					if rt.task_id == self.jobMgr.ready_queue[i].task_id: 
						isRunning = True
						break
				if isRunning:
					continue
				
				no_rsrc = True
				for gpu in gpu_list:
					if gpu.get_status() is not gpu_status_t.FULL:
						no_rsrc = False
				if 	no_rsrc is True:
					break
				
				job = None
				if self.policy_name == "mg-jm":
					job = job_migration(task_list, gpu_list, self.jobMgr.ready_queue[i], [])
				elif self.policy_name == "mg-offline":
					job = self.mgOfflineHelper(self.jobMgr.ready_queue[i])
				if job is not None:
					self.jobMgr.ready_queue[i] = job.clone()
					self.jobMgr.execute(self.jobMgr.ready_queue[i])
					rev_index.append(i)
			for ri in sorted(rev_index, reverse=True):
				del self.jobMgr.ready_queue[ri]
			
			# predict power
			# print("instant power = {}".format(compute_tick_energy(gpu_list)))
			self.jobMgr.energy_pred += compute_tick_energy(gpu_list)
			sys_time.increment_sys_time()

	def shutDown(self) -> None:
		printWrapper(self.verbose, "Total number of released jobs: {}".format(self.jobMgr.released))
		printWrapper(self.verbose, "Total number of missed jobs: {}".format(self.jobMgr.missed))
		printWrapper(self.verbose, "Total predicted energy: {}".format(self.jobMgr.energy_pred))
		if self.verbose is False:
			self.write_results()

	def write_results(self):
		# filename format: example/taskset_08022022/set_uXX/YYY.csv
		a,b,c,d = self.filename.split("/")
		ux = c[5:]
		fn = "output/taskset_08022022/" + self.policy_name + "/set_u" + str(ux).zfill(2) + ".csv"
		# write: filename,released,missed,energy
		data = [self.filename, self.jobMgr.released, self.jobMgr.missed, self.jobMgr.energy_pred]
		with open(fn, "a", encoding="utf-8", newline="") as f:
			writer = csv.writer(f)
			writer.writerow(data)


