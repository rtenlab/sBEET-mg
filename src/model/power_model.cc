#include "common/include/messages.h"

#include "power_model.h"
#include "gpu_model.h"
#include "gpu_slot_model.h"

float Power::compute_energy_constant(float *tw, float power) {
	return power * (tw[1] - tw[0]);
}

float Power::predict_system_power(std::vector<GPU> gpu_list, float tp) {
	float power = 0;
	for (int i = 0; i < gpu_list.size(); i++) {
		if (gpu_list[i].get_status_tp(tp) != IDLE) {
			for (int j = 0; j < gpu_list[i].slots.size(); j++) {
				if (!gpu_list[i].slots[j].empty_tp(tp)) {
					app_t _aid = gpu_list[i].slots[j].get_app_tp(tp);
					unsigned int sm = gpu_list[i].get_sm_slot_tp(tp, j);
					power += gpu_list[i].power_dynamic[_aid] * sm;
				}
			}
			// add power consumption from idle SMs
			power += (gpu_list[i].max_sm - gpu_list[i].get_act_sm_tp(tp)) * gpu_list[i].power_idle;
		}
		power += gpu_list[i].power_static;
	}
	return power;
}

float Power::predict_system_energy(std::vector<GPU> gpu_list, float *tw) {
	float energy = 0.0f;
	for (int t = int(tw[0]); t < int(tw[1]) + 1; t++) {
		energy += predict_system_power(gpu_list, float(t));
	}
	return energy;
}

float Power::compute_tick_energy(std::vector<GPU> gpu_list) {
	float energy = 0.0f;
	for (auto g : gpu_list) {
		float power = g.power_static;
		if (g.get_status() != IDLE) {
			for (auto s : g.slots) {
				if (!s.empty_()) {
					power += g.power_dynamic[s.job->app_id] * s.get_sm();
				}
			}
			power += g.power_idle * (g.max_sm - g.get_act_sm());
		}
		energy += power / 1000;
	}
	return energy;
}