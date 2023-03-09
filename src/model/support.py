from dataclasses import dataclass
from typing import List

dict_rtx3070 = {
	"max_sm": 46,
	"power_static": 46,
	"power_idle": 0.445,
	"power_dynamic": [3.77, 1.63, 1.14, 1.67, 0.98, 1.08, 3.18, 0.91],
	"ghd": [2.5, 4, 1.5, 1, 4.5, 2, 1.5, 1],
	"gdh": [1, 1, 1, 1, 1, 1, 1, 1]
}

dict_t400 = {
	"max_sm": 6,
	"power_static": 8,
	"power_idle": 0.652,
	"power_dynamic": [2.06, 0.98, 0.81, 1.15, 1.07, 1.29, 2.06, 1.19],
	"ghd": [13, 1, 3, 12, 6, 4.5, 3, 2.5],
	"gdh": [1.5, 1, 1.5, 4, 5.5, 1, 1.5, 1]
}

@dataclass
class gpu_simple_t:
	gpu_id: int
	name: str

GPUs = []
GPUs.append(gpu_simple_t(0, "rtx3070"))
GPUs.append(gpu_simple_t(1, "rtx3070"))
GPUs.append(gpu_simple_t(2, "t400"))

# unit: ms, or tick
class SysTime:
	def __init__(self):
		self.sys_tick: float = 0.0

	def init_sys_time(self):
		self.sys_tick = 0.0
	
	def get_sys_time(self) -> float:
		return self.sys_tick

	def increment_sys_time(self) -> None:
		self.sys_tick += 1

	def convert_to_string(self) -> str:
		s = int(self.sys_tick / 1000)
		ms = int(self.sys_tick % 1000)
		return str(s).zfill(3) + "." + str(ms).zfill(3)

sys_time = SysTime()
	