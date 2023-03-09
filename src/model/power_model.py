from typing import List

from common.define import *
from src.model.gpu_model import GPU

def compute_energy_constant(tw: List[float], power: float) -> float:
	return power * (tw[1] - tw[0])

def predict_system_power(gpu_list: List[GPU], tp: float) -> float:
	power = 0.0
	for g in gpu_list:
		if g.get_status_tp(tp) != gpu_status_t.IDLE:
			for s in g.slots:
				if s.empty_tp(tp) is False:
					_aid = s.get_app_tp(tp)
					_sm = g.get_sm_slot_tp(tp, s.id)
					power += g.power_dynamic[_aid] * _sm
			power += (g.max_sm - g.get_act_sm_tp(tp)) * g.power_idle
		power += g.power_static
	return power

def predict_system_energy(gpu_list: List[GPU], tw: List[float]) -> float:
	energy = 0.0
	for t in range(round(tw[0]), round(tw[1]) + 1):
		energy += predict_system_power(gpu_list, float(t))
	return energy

def compute_tick_energy(gpu_list: List[float]) -> float:
	energy = 0.0
	for g in gpu_list:
		pd = 0.0
		pidle = 0.0
		ps =  g.power_static
		if g.get_status() != gpu_status_t.IDLE:
			for s in g.slots:
				if s.empty_() is False:
					pd += g.power_dynamic[s.job.app_id] * s.get_sm()
			pidle += g.power_idle * (g.max_sm - g.get_act_sm())
		energy += (ps + pd + pidle) / 1000
	return energy