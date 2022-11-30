#include "gpu_slot_model.h"
#include "support.h"

GPU_JobSlot::GPU_JobSlot() {
	this->is_empty = true;
	this->sm = 0;
	this->sm_map = 0x0;
	this->job = nullptr;
	waitlist.reserve(20);
}

GPU_JobSlot::~GPU_JobSlot() {}

void GPU_JobSlot::assign_job_slot(Job *__job){
	this->job = __job;
	update_sm_map(__job->get_sm_map());
	is_empty = false;
}

void GPU_JobSlot::delete_job_slot() {
	this->job = nullptr;
	update_sm_map(0x0);
	is_empty = true;
	clear_waitlist();
}

void GPU_JobSlot::update_sm_map(uint64_t _sm_map) {
	this->sm_map = _sm_map;
	this->sm = get_sm_by_map(_sm_map);
}

/* Return the OCCUPIED number of SMs at tp */
const unsigned int GPU_JobSlot::get_sm_tp(float tp) const {
	if (!empty_()) {
		if (tp < this->job->gpu_end_time) {
			return this->job->get_sm();
		}
	}
	for (auto it : this->waitlist) {
		if (tp < it.gpu_end_time && tp >= it.gpu_start_time) {
			return it.get_sm();
		}
	}
	return 0;
}

const app_t GPU_JobSlot::get_app_tp(float tp) const {
	if (!empty_()) {
		if (tp < this->job->gpu_end_time) {
			return this->job->app_id;
		}
	}
	bool failed = true;
	for (auto it : this->waitlist) {
		if (tp < it.gpu_end_time && tp >= it.gpu_start_time) {
			return it.app_id;
		}
	}
	if (failed) {
		throw std::runtime_error("const app_t GPU_JobSlot::get_app_tp(float tp) const failed");
	}
}