#include <bitset>
#include <stdexcept>
#include <assert.h>

#include "common/include/defines.h"
#include "gpu_model.h"
#include "support.h"


GPU::GPU(unsigned int _gpu_id, std::string _name) {
	this->gpu_id = _gpu_id;
	this->name = _name;
	this->util = 0.0f;
	if (name == "t400") {
		this->max_sm = T400::max_sm;
		this->total_sm = T400::total_sm;
		this->power_static = T400::power_static;
		this->power_idle = T400::power_idle;
		this->power_dynamic = T400::power_dynamic;
		// this->ghd = T400::ghd;
		// this->gdh = T400::gdh;
	}else if (name == "rtx3070") {
		this->max_sm = RTX3070::max_sm;
		this->total_sm = RTX3070::total_sm;
		this->power_static = RTX3070::power_static;
		this->power_idle = RTX3070::power_idle;
		this->power_dynamic = RTX3070::power_dynamic;
		// this->ghd = RTX3070::ghd;
		// this->gdh = RTX3070::gdh;
	}

	this->total_sm_map = (1 << this->total_sm) - 1;
	this->act_sm = 0;
	this->act_sm_map = 0x0;
	this->slots.reserve(2);
	for (int i = 0; i < partition_num; i++) {
		this->slots.push_back(GPU_JobSlot());
	}
}

GPU::~GPU() {}

int GPU::assign_slot(Job *__job, int sid) {
	if (sid != -1) { // assigned to a specific slot
		if (this->slots[sid].empty_()) {
			this->slots[sid].assign_job_slot(__job);
			add_sm_into_use(__job->get_sm_map());
			return sid;
		}else {
			std::cout << "Assigning an anavailable GPU slot! Automatically select one instead." << std::endl;
		}
	}
	for (int i = 0; i < partition_num; i++) {
		if (this->slots[i].empty_()) {
			this->slots[i].assign_job_slot(__job);
			add_sm_into_use(__job->get_sm_map());
			return i;
		}
	}
}

int GPU::assign_slot_tp(Job *__job, float tp) {
	bool failed = true;
	for (int i = 0; i < partition_num; i++) {
		if (this->slots[i].empty_tp(tp)) {
			this->slots[i].push_to_waitlist(*__job);
			return i;
		}
	}
	if (failed) {
		throw std::runtime_error("int GPU::assign_slot_tp(Job *__job, float tp) failed.");
	}
}

void GPU::reset_slot(int slot_id) {
	rev_sm_from_use(this->slots[slot_id].sm_map);
	this->slots[slot_id].delete_job_slot();
}

void GPU::reset_slot_waitlist(int slot_id) {
	this->slots[slot_id].clear_waitlist();
}

/* when the GPU is partially occupied, get the slot that is active */
unsigned int GPU::get_active_slot() {
	assert(get_status() == ACTIVE);
	if (!slots[0].empty_()) 
		return 0;
	else if (!slots[1].empty_()) 
		return 1;
}

/* when the gpu is full */
unsigned int GPU::get_faster_slot() {
	assert(get_status() == FULL);
	unsigned int slot_id = -1;
	if (!this->slots[0].empty_() && !this->slots[1].empty_()) {	// (1) two running jobs
		if (this->slots[0].job->gpu_end_time < this->slots[1].job->gpu_end_time)
			slot_id = 0;
		else
			slot_id = 1;
	} else {  // (2) one job is using all SMs, return it
		if (!this->slots[0].empty_()) slot_id = 0;
		else slot_id = 1;
	}
	return slot_id;
}

float GPU::calc_power_static() {
	return this->power_static;
}

float GPU::calc_power_idle() {
	if (get_status() == IDLE) return 0;
	else return (this->total_sm - this->act_sm) * this->power_idle;
}

float GPU::calc_power_total() {
	if (get_status() == IDLE)
		return calc_power_static();
	else {
		float power = 0;
		power += slots[0].calc_power_d(power_dynamic, slots[0].job) + slots[1].calc_power_d(power_dynamic, slots[1].job);
		power += calc_power_idle() + calc_power_static();
		return power;
	}
}

unsigned int GPU::get_sm_slot_tp(float tp, unsigned int slot_id) {
	return this->slots[slot_id].get_sm_tp(tp);	
}

unsigned int GPU::get_act_sm_tp(float tp) {
	unsigned int sm0, sm1;
	unsigned int ret = 0;
	for (int i = 0; i < this->slots.size(); i++) {
		ret += this->slots[i].get_sm_tp(tp);
	}
	return ret;
}

uint64_t GPU::gen_sm_map(unsigned int sm) {
	uint64_t ret = 0;
	uint64_t usable_sm_map = total_sm_map & (~this->act_sm_map);
	int ex = 0;

	int cnt = 0;
	while (sm > 0) {
		if (usable_sm_map & 1) {
			sm--;
			ret |= 1 << cnt;
		}
		usable_sm_map >>= 1;
		cnt++;
	}
	return ret;
}

void GPU::add_sm_into_use(uint64_t _sm_map) {
	this->act_sm_map |= _sm_map;
	this->act_sm = get_sm_by_map(this->act_sm_map);
}

void GPU::rev_sm_from_use(uint64_t _sm_map) {
	this->act_sm_map &= ~_sm_map;
	this->act_sm = get_sm_by_map(this->act_sm_map);
}