#include <algorithm>

#include "common/include/messages.h"
#include "common/include/defines.h"
#include "task_model.h"
#include "support.h"

std::string workspace_folder = "/home/wyd/Documents/rten/sbeet_mg/";
std::string gpu_names[2] = {"rtx3070", "t400"};
std::string app_names[8] = {"mmul", "stereodisparity", "hotspot", "dxtc", "bfs", "hist", "mmul2", "hist2"};


// YIDI: assuming the `gpu_id` is of the same order as my physical GPU id
const float get_ghd(app_t app_id, unsigned int gpu_id) {
	if (gpu_id == 0) { // rtx 3070
		return RTX3070::ghd[app_id];
	}else if (gpu_id == 1) { // t400
		return T400::ghd[app_id];
	}
}

const float get_gdh(app_t app_id, unsigned int gpu_id) {
	if (gpu_id == 0) { // rtx 3070
		return RTX3070::gdh[app_id];
	}else if (gpu_id == 1) { // t400
		return T400::gdh[app_id];
	}
}

bool energy_efficiency_comparator (const task_param_t &a, const task_param_t &b) {
	return a.energy < b.energy;
}

Task::Task(app_t app_id, float period, float phi, float ref_util) {
	this->app_id = app_id;
	this->period = period;
	this->phi = phi;
	this->ref_util = ref_util;
}

Task::~Task() {}

void Task::get_best_energy_efficiency(unsigned int _gpu_id, unsigned int* _sm, float* energy, std::vector<float>& wcets) {
	std::string fn = workspace_folder + "profiles/gpu_" + gpu_names[_gpu_id] + "/" + app_names[this->app_id] + ".csv";
	std::vector<float> energy_arr = read_csv_column(fn, 6); // energy_in_window
	wcets = read_csv_column(fn, 3); // Max(ms)
	float min_e = std::numeric_limits<float>::max();
	unsigned int index = 0;
	for (int i = 0; i < sm_limit[_gpu_id]; i++) {
		if (energy_arr[i] < min_e) {
			min_e = energy_arr[i];
			index = i;
		}
	}
	*_sm = index + 1;
	*energy = min_e;
}

unsigned int Task::get_mopt(unsigned int _gpu_id) const{
	unsigned int idx = search_in_gpu_rankings(_gpu_id);
	unsigned int mopt = gpu_rankings[idx].mopt;
	return mopt;
}

// use `sm_limit`, not `reference_sm` since here is the real utilization of a task, not the reference utilization
float Task::get_util(unsigned int _gpu_id) const {
	std::string fn = workspace_folder + "profiles/gpu_" + gpu_names[_gpu_id] + "/" + app_names[this->app_id] + ".csv";
	std::vector<float> wcet_arr = read_csv_column(fn, 3); // Max(ms)
	float wcet_sum = 0.0f;
	for (int i = 0; i < sm_limit[_gpu_id]; i++) {
		wcet_sum += wcet_arr[i];
	}
	return float((wcet_sum / sm_limit[_gpu_id] + (get_ghd(this->app_id, _gpu_id) + get_gdh(this->app_id, _gpu_id))) / this->period);
}

float Task::get_util_mopt(unsigned int _gpu_id) const {
	unsigned int mopt = get_mopt(_gpu_id);
	std::string fn = workspace_folder + "profiles/gpu_" + gpu_names[_gpu_id] + "/" + app_names[this->app_id] + ".csv";
	std::vector<float> wcet_arr = read_csv_column(fn, 3); // Max(ms)
	float wcet = wcet_arr[mopt - 1];
	return (wcet + get_ghd(this->app_id, _gpu_id) + get_gdh(this->app_id, _gpu_id)) / this->period; 
}

void Task::init_gpu_rankings() {
	for(int i = 0; i < gpu_num; i++) {
		unsigned int _mopt;
		float _energy;
		std::vector<float> _wcets;
		get_best_energy_efficiency(i, &_mopt, &_energy, _wcets);
		gpu_rankings.push_back({(unsigned int)i, _mopt, _energy, _wcets});
	}
	std::sort(gpu_rankings.begin(), gpu_rankings.end(), energy_efficiency_comparator);
}

/* runtime functions */
void Task::_taskInit() {
	workload->taskInit();
}

void Task::_taskInitDevice(unsigned int _id) {
	workload->taskInitDevice(_id);
}

std::mutex shared_mutex;

void Task::_taskComplete() {
	shared_mutex.lock();
	this->workload->taskFinish();
	shared_mutex.unlock();
}
