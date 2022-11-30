#ifndef __TASK_MODEL_H__
#define __TASK_MODEL_H__

#include <vector>
#include <mutex>
#include <stdexcept>

#include "app/base_task.h"

enum app_t {MMUL, STEREODISPARITY, HOTSPOT, DXTC, BFS, HIST, MMUL2, HIST2};

/* gpu_id, energy, mopt, wcet */
struct task_param_t {
	unsigned int gpu_id;
	unsigned int mopt;
	float energy;
	std::vector<float> wcets; // wcets across different sms on `gpu_id`
};

bool energy_efficiency_comparator (const task_param_t &a, const task_param_t &b);

extern std::mutex shared_mutex;

const float get_ghd(app_t app_id, unsigned int gpu_id);
const float get_gdh(app_t app_id, unsigned int gpu_id);

class Task {
public:
	Task(app_t app_id, float period, float phi, float ref_util);
	~Task();

	/* variables */
	unsigned int task_id;
	app_t app_id;
	float period;
	float phi;
	float ref_util;
	unsigned int mopt;
	unsigned int gpu_id;

	BASE_TASK *workload;

	/* derived variables */
	std::vector<task_param_t> gpu_rankings; // should be obtained at the offline stage, struct of { energy, mopt, energy, wcets } in ascending order of energy

	/* functions */
	void init_gpu_rankings(); // init the gpu_rankings vector, offline only
	unsigned int get_mopt(unsigned int _gpu_id) const;
	float get_util(unsigned int _gpu_id) const; // get the oevrall util of the task on the gpu_id, offline only
	float get_util_mopt(unsigned int _gpu_id) const;
	inline const unsigned int search_in_gpu_rankings(unsigned int _gpu_id) const;

	/* runtime functions */
	void _taskInit();
	void _taskInitDevice(unsigned int);
	void _taskComplete();

private:

	void get_best_energy_efficiency(unsigned int _gpu_id, unsigned int *_sm, float *energy, std::vector<float>& wcets); // return the mopt and its corresponding energy of a certain GPU
};

/* Return the index in gpu_rankings  */
inline const unsigned int Task::search_in_gpu_rankings(unsigned int _gpu_id) const {
	bool failed = true;
	for (int i = 0; i < this->gpu_rankings.size(); i++) {
		if (_gpu_id == this->gpu_rankings[i].gpu_id) {
			return i;
		}
	}
	if (failed) {
		throw std::runtime_error("inline const unsigned int Task::search_in_gpu_rankings(unsigned int _gpu_id) const failed.");
	}
}

#endif // __TASK_MODEL_H__