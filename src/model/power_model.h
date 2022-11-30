#ifndef __POWER_MODEL_H__
#define __POWER_MODEL_H__

#include "gpu_model.h"

namespace Power {
	/* compute the energy in a time window if the power is constant */
	float compute_energy_constant(float *tw, float power);
	/* predict the system power of one timepoint and add them up to predict the energy consumption in a window */
	float predict_system_power(std::vector<GPU> gpu_list, float tp);
	/* input: time window */
	float predict_system_energy(std::vector<GPU> gpu_list, float *tw);
	/* Call on the runtime to estimate the overall energy consumption */
	float compute_tick_energy(std::vector<GPU> gpu_list);
};

#endif // !__POWER_MODEL_H__