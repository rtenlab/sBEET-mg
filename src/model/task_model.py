from dataclasses import dataclass
import ctypes
import os, sys
import pandas as pd
import operator
from typing import List, Tuple

from src.model.support import dict_rtx3070, dict_t400, GPUs
from common.define import *

@dataclass
class task_param_t:
	gpu_id: int
	mopt: int
	energy: float
	wcets: list

# assuming gpu_id == idx
def get_ghd(app_id: int, gpu_id: int) -> float:
	if GPUs[gpu_id].name == "rtx3070" or GPUs[gpu_id].name == "RTX3070": 
		return dict_rtx3070["ghd"][app_id]
	elif GPUs[gpu_id].name == "t400" or GPUs[gpu_id].name == "T400":
		return dict_t400["ghd"][app_id]

def get_gdh(app_id: int, gpu_id: int) -> float:
	if GPUs[gpu_id].name == "rtx3070" or GPUs[gpu_id].name == "RTX3070": 
		return dict_rtx3070["gdh"][app_id]
	elif GPUs[gpu_id].name == "t400" or GPUs[gpu_id].name == "T400":
		return dict_t400["ghd"][app_id]
	
class Task:
	def __init__(self, app_id, period, phi, ref_util):
		self.app_id: int = app_id
		self.period: float = period
		self.phi: float = phi
		self.ref_util: float = ref_util

		self.gpu_rankings: List[task_param_t] = list()

	def init_gpu_rankings(self) -> None:
		for i in range(len(GPUs)):
			_mopt, _energy, _wcets = self.get_best_energy_efficiency(GPUs[i].gpu_id)
			self.gpu_rankings.append(task_param_t(GPUs[i].gpu_id, _mopt, _energy, _wcets))
		self.gpu_rankings.sort(key=lambda x: x.energy)

	def get_mopt(self, _gpu_id: int) -> int:
		idx = self.search_in_gpu_rankings(_gpu_id)
		return self.gpu_rankings[idx].mopt

	# returns the real utlization, not the reference utilization
	def get_util(self, _gpu_id: int) -> float:
		fn = os.path.join(database_folder, "gpu_" + GPUs[_gpu_id].name, app_names[self.app_id] + ".csv")
		df = pd.read_csv(fn)
		wcets = df["Max(ms)"].to_list()
		lm = sm_limit[_gpu_id]
		return (sum(wcets[:lm]) / lm + get_ghd(self.app_id, _gpu_id) + get_gdh(self.app_id, _gpu_id)) / self.period

	# dedicated for offline alg
	def get_util_mopt(self, _gpu_id: int) -> float:
		mopt = self.get_mopt(_gpu_id)
		fn = os.path.join(database_folder, "gpu_" + GPUs[_gpu_id].name, app_names[self.app_id] + ".csv")
		df = pd.read_csv(fn)
		wcet = df["Max(ms)"].to_list()[mopt-1]
		return (wcet + get_ghd(self.app_id, _gpu_id) + get_gdh(self.app_id, _gpu_id)) / self.period

	def search_in_gpu_rankings(self, _gpu_id: int) -> int:
		ret = -1
		for ret in range(len(self.gpu_rankings)):
			if _gpu_id == self.gpu_rankings[ret].gpu_id:
				return ret

	def get_best_energy_efficiency(self, _gpu_id: int) -> Tuple[int, float, List[float]]:
		fn = os.path.join(database_folder, "gpu_" + GPUs[_gpu_id].name, app_names[self.app_id] + ".csv")
		df = pd.read_csv(fn)
		energy_arr = df["energy_in_window"].to_list()
		wcets = df["Max(ms)"].to_list()
		lm = sm_limit[_gpu_id]
		sm = energy_arr.index(min(energy_arr[:lm])) + 1
		return sm, min(energy_arr[:lm]), wcets

# todo: such as taskInit, taskComplete, etc, add in the future if needed
