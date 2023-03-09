import os 
import pandas as pd

from common.define import *
from src.model.task_model import get_ghd, get_gdh
from src.model.support import GPUs

# No need to maintain the feilds related to sm_map
class Job:
	def __init__(self, task_id: int, app_id: int, gpu_id: int, release_time: float, deadline: float):
		self.gpu_id: int = gpu_id
		self.task_id: int = task_id
		self.app_id: int = app_id
		self.release_time: float = release_time
		self.deadline: float = deadline

		self.start_time = 0.0
		self.gpu_start_time = 0.0
		self.gpu_end_time = 0.0
		self.end_time = 0.0
		
		self.sm = 0
	
	def get_gpu_wcet(self): # return the wcet on the delegated GPU
		# todo: it should be a characteristic of a task
		for i in range(len(GPUs)):
			if self.gpu_id == GPUs[i].gpu_id:
				fn = os.path.join(database_folder, "gpu_" + GPUs[i].name, app_names[self.app_id] + ".csv")
				df = pd.read_csv(fn)
				wcets = df["Max(ms)"].to_list()
				return wcets[self.sm - 1]


	def update_timing(self, tp: float) -> None:
		self.start_time = tp
		self.gpu_start_time = self.start_time + get_ghd(self.app_id, self.gpu_id)
		self.gpu_end_time = self.gpu_start_time + self.get_gpu_wcet()
		self.end_time = self.gpu_end_time + get_gdh(self.app_id, self.gpu_id)

	def update_config(self, _gpu_id: int, _sm: int, tp: float) -> None:
		self.gpu_id = _gpu_id
		self.update_sm(_sm)
		self.update_timing(tp)

	def check_deadline(self) -> bool:
		return self.end_time <= self.deadline

	def get_sm(self) -> int:
		return self.sm

	def update_sm(self, _sm: int) -> None:
		self.sm = _sm

	def clone(self):
		job_ = Job(self.task_id, self.app_id, self.gpu_id, self.release_time, self.deadline)
		job_.start_time = self.start_time
		job_.gpu_start_time = self.gpu_start_time
		job_.gpu_end_time = self.gpu_end_time
		job_.end_time = self.end_time
		job_.sm = self.sm
		job_.app_id = self.app_id
		return job_