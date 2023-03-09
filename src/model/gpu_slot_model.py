from typing import List

from src.model.job_model import Job

class GPU_JobSlot:
	def __init__(self, _sid):
		self.job: Job = None # the methods will not modify this feild
		self.is_empty: bool = True
		self.sm: int = 0 # occupied sm
		self.waitlist = []
		self.id = _sid

	def empty_(self) -> bool:
		return self.is_empty
	
	def empty_tp(self, tp: float) -> bool:
		if self.empty_() is False:
			if tp < self.job.gpu_end_time:
				return False
		for it in self.waitlist:
			if tp < it.gpu_end_time and tp >= it.gpu_start_time:
				return False
		return True
	
	def get_sm(self) -> int:
		return self.sm

	def get_sm_tp(self, tp: float) -> int:
		if self.empty_() is False:
			if tp < self.job.gpu_end_time:
				return self.job.get_sm()
		for it in self.waitlist:
			if tp < it.gpu_end_time and tp >= it.gpu_start_time:
				return it.get_sm()
		return 0

	def update_sm(self, _sm: int):
		self.sm = _sm
	
	def push_to_waitlist(self, _job: Job) -> None:
		self.waitlist.append(_job)
	
	def clear_waitlist(self) -> None:
		self.waitlist.clear()

	def assign_job_slot(self, _job: Job) -> None:
		self.job = _job
		self.update_sm(_job.get_sm())
		self.is_empty = False

	def delete_job_slot(self) -> None:
		self.job = None
		self.update_sm(0)
		self.is_empty = True
		self.clear_waitlist()

	def get_app_tp(self, tp: float) -> int:
		if self.empty_() is False:
			if tp < self.job.gpu_end_time:
				return self.job.app_id
		for it in self.waitlist:
			if tp < it.gpu_end_time and tp >= it.gpu_start_time:
				return it.app_id

	def calc_power_d(self, pd_ls: List[float], _job: Job) -> float:
		if _job == None:
			return 0
		return _job.get_sm() * pd_ls[_job.app_id]
