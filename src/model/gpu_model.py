from typing import List

from src.model.gpu_slot_model import GPU_JobSlot
from common.define import *
from src.model.support import dict_rtx3070, dict_t400
from src.model.job_model import Job

class GPU:
	def __init__(self, _gpu_id: int, name: str):
		self.name = name
		self.gpu_id = _gpu_id
		self.util: float = 0.0
		self.total_sm: int = sm_limit[_gpu_id]
		if self.name == "t400" or self.name == "T400":
			self.max_sm = dict_t400["max_sm"]
			self.power_static = dict_t400["power_static"]
			self.power_idle = dict_t400["power_idle"]
			self.power_dynamic = dict_t400["power_dynamic"]
		elif self.name == "rtx3070" or self.name == "RTX3070":
			self.max_sm = dict_rtx3070["max_sm"]
			self.power_static = dict_rtx3070["power_static"]
			self.power_idle = dict_rtx3070["power_idle"]
			self.power_dynamic = dict_rtx3070["power_dynamic"]

		self.act_sm: int = 0
		self.slots = [GPU_JobSlot(0), GPU_JobSlot(1)]

	def get_status(self) -> gpu_status_t:
		if self.slots[0].empty_() and self.slots[1].empty_():
			return gpu_status_t.IDLE
		elif (self.slots[0].get_sm() + self.slots[1].get_sm() == self.total_sm) or (self.slots[0].empty_() is False and self.slots[1].empty_() is False):
			return gpu_status_t.FULL
		else:
			return gpu_status_t.ACTIVE

	def get_status_tp(self, tp: float) -> gpu_status_t:
		if self.slots[0].empty_tp(tp) and self.slots[1].empty_tp(tp):
			return gpu_status_t.IDLE
		elif (self.slots[0].get_sm_tp(tp) + self.slots[1].get_sm_tp(tp) == self.total_sm) or (self.slots[0].empty_tp(tp) is False and self.slots[1].empty_tp(tp) is False):
			return gpu_status_t.FULL
		else:
			return gpu_status_t.ACTIVE

	def get_act_sm(self) -> int:
		return self.slots[0].get_sm() + self.slots[1].get_sm()

	def get_rem_sm(self) -> int:
		return self.total_sm - self.get_act_sm()

	def get_act_sm_tp(self, tp: float) -> int:
		self.slots[0].get_sm_tp(tp) + self.slots[1].get_sm_tp(tp)

	def add_sm_into_use(self, _sm: int):
		self.act_sm += _sm

	def rm_sm_from_use(self, _sm: int):
		self.act_sm -= _sm

	def assign_slot(self, _job: Job, sid: int) -> int:
		if sid != -1:
			if self.slots[sid].empty_():
				self.slots[sid].assign_job_slot(_job)
				self.add_sm_into_use(_job.get_sm())
				return sid
			else:
				print("Warning! Assigning an anavailable GPU slot! Automatically select one instead.")
		for i in range(partition_num):
			if self.slots[i].empty_():
				self.slots[i].assign_job_slot(_job)
				self.add_sm_into_use(_job.get_sm())
				return i

	def assign_slot_tp(self, _job: Job, tp: float) -> int:
		for i in range(partition_num):
			if self.slots[i].empty_tp(tp):
				self.slots[i].push_to_waitlist(_job)
				return i

	def reset_slot(self, sid: int):
		self.rm_sm_from_use(self.slots[sid].get_sm())
		self.slots[sid].delete_job_slot()

	def reset_slot_waitlist(self, sid: int): # todo: can it be merged into reset_slot()?
		self.slots[sid].clear_waitlist()
	
	def get_active_slot(self) -> int:
		if self.get_status() == gpu_status_t.ACTIVE:
			if self.slots[0].empty_() is False:
				return 0
			elif self.slots[1].empty_() is False:
				return 1
	
	def get_faster_slot(self) -> int: # only valid when the GPU is full
		sid = -1
		if self.get_status() == gpu_status_t.FULL:
			if self.slots[0].empty_() is False and self.slots[1].empty_() is False:
				if self.slots[0].job.gpu_end_time < self.slots[1].job.gpu_end_time:
					sid = 0
				else:
					sid = 1
			elif self.slots[0].empty_() is False:
				sid = 0
			elif self.slots[1].empty_() is False:
				sid = 1
		return sid

	def calc_power_static(self) -> float:
		return self.power_static
	
	def calc_power_idle(self) -> float:
		if self.get_status() == gpu_status_t.IDLE:
			return 0
		else:
			return (self.max_sm - self.act_sm) * self.power_idle # todo: fix in hw v1 code

	def calc_power_total(self) -> float:
		if self.get_status() == gpu_status_t.IDLE:
			return self.calc_power_static()
		else:
			power = self.calc_power_static() + self.calc_power_idle()
			for i in range(partition_num):
				power += self.slots[i].calc_power_d(self.power_dynamic, self.slots[i].job)
			return power
	
	def get_sm_slot_tp(self, tp: float, sid: int) -> int:
		return self.slots[sid].get_sm_tp(tp)

	def get_act_sm_tp(self, tp: float) -> int:
		ret = 0
		for i in range(partition_num):
			ret += self.slots[i].get_sm_tp(tp)
		return ret


