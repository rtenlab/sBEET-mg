from dataclasses import dataclass
from typing import List, Set, Union
import pandas as pd

from src.model.task_model import Task
from src.model.job_model import Job
from src.model.gpu_model import GPU
from src.model.support import GPUs
from src.model.power_model import *
from src.model.support import sys_time
from common.define import *
from common.runtime import *
from src.mg import mg
from src.b_ofl import b_ofl

''' Common stages '''
def loadTaskset(filename) -> None:
	df = pd.read_csv(filename)
	app_list = df["app"].to_list()
	period_list = df["period"].to_list()
	phi_list = df["phi"].to_list()
	rutil_list = df["ref_util"].to_list()
	for i in range(len(app_list)):
		task_list.append(Task(app_list[i], period_list[i], phi_list[i], rutil_list[i]))
	for i in range(len(task_list)):
		task_list[i].init_gpu_rankings()

def initGPUs() -> None:
	for gg in GPUs:
		gpu_list.append(GPU(gg.gpu_id, gg.name))

def Run(_filename: str, _duration: float, _policy_name: str, _verbose: bool, _emode: bool):
	initGPUs()

	loadTaskset(_filename)

	mgIns = mg(_filename, _verbose, _duration, _policy_name, _emode)
	boIns = b_ofl(_filename, _verbose, _duration, _policy_name, _emode)

	if _policy_name == "mg-jm" or _policy_name == "mg-offline":
		mgIns.taskDistribution()
		printTaskset(_verbose, _policy_name, task_list)
		mgIns.schedulerOnline()
		mgIns.shutDown()
	elif _policy_name == "lcf" or _policy_name == "bcf" or _policy_name == "ld":
		boIns.taskDistribution()
		printTaskset(_verbose, _policy_name, task_list)
		boIns.schedulerOnline()
		boIns.shutDown()
	else:
		print("Unknown policy name. Abort!")
		exit(1)

	# destroy global vars
	task_list.clear()
	gpu_list.clear()