import os, sys
from enum import Enum

workplace_folder = ""
database_folder = os.path.join(workplace_folder, "database")

sys.path.append("../src/model")
sys.path.append("../src/algorithm")

class App(Enum):
	MMUL = 0
	STEREODISPARITY = 1
	HOTSPOT = 2
	DXTC = 3
	BFS = 4
	HIST = 5
	MMUL2 = 6
	HIST2 = 7

app_names = ["mmul", "stereodisparity", "hotspot", "dxtc", "bfs", "hist", "mmul2", "hist2"]

class gpu_status_t(Enum):
	IDLE = 0
	FULL = 1
	ACTIVE = 2

deadline_factor = 0.5
partition_num = 2
sm_limit = [12, 12, 6] # remember to modify GPUs in support.py