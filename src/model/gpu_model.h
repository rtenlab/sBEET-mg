#ifndef __GPU_MODEL_H__
#define __GPU_MODEL_H__

#include <string>
#include <vector>
#include "gpu_slot_model.h"
#include "worker.h"

enum gpu_status_t {IDLE, ACTIVE, FULL};

class GPU {
public:
	GPU(unsigned int _gpu_id, std::string name);
	~GPU();

	/* variables */
	std::string name;
	unsigned int total_sm; // const, the total number of SMs allowed to use
	unsigned int max_sm; // const, the max number of SMs the GPU has
	uint64_t total_sm_map;	// const
	unsigned int gpu_id;

	float power_static;
	float power_idle;
	float* power_dynamic; // index: app_id
	float util;

	/* functions */
	inline const gpu_status_t get_status() const;
	inline const gpu_status_t get_status_tp(float tp) const;

	int assign_slot(Job *__job, int sid); // return slot index
	int assign_slot_tp(Job *__job, float tp);

	void reset_slot(int slot_id);
	void reset_slot_waitlist(int slot_id);

	/* generate a valid SM assignation according to the status of the GPU */
	uint64_t gen_sm_map(unsigned int sm);

	unsigned int get_active_slot(); // valid only when the gpu is partially occupied
	unsigned int get_faster_slot(); // valid only when the gpu is full

	/* power related */
	float calc_power_total(); // calc the instant power of the whole GPU

	inline unsigned int get_act_sm();
	inline unsigned int get_rem_sm();
	/* return the number of SMs a slot occupies */
	unsigned int get_sm_slot_tp(float tp, unsigned int slot_id);
	/* Get number of active SMs at tp */
	unsigned int get_act_sm_tp(float tp);

	std::vector<GPU_JobSlot> slots;

	/* runtime variables */
	WorkerPool *worker_pool;

protected:
	void add_sm_into_use(uint64_t _sm_map);	// update all the SM related fields
	void rev_sm_from_use(uint64_t _sm_map);	// update all the SM related fields

private:
	unsigned int act_sm; // active sm in use
	uint64_t act_sm_map;

	/* power related */
	float calc_power_idle(); // calc the instant power of idle SMs
	float calc_power_static();
};

inline const gpu_status_t GPU::get_status() const {
	if (this->slots[0].empty_() && this->slots[1].empty_()) {
		return IDLE;
	}else if ((this->slots[0].get_sm() + this->slots[1].get_sm() == this->total_sm) || (!this->slots[0].empty_() && !this->slots[1].empty_())) {
		return FULL;
	}else {
		return ACTIVE;
	}
}

inline const gpu_status_t GPU::get_status_tp(float tp) const {
	if (this->slots[0].get_sm_tp(tp) + this->slots[1].get_sm_tp(tp) == this->total_sm) {
		return FULL;
	}else if (this->slots[0].empty_tp(tp) && this->slots[1].empty_tp(tp)) {
		return IDLE;
	}else {
		return ACTIVE;
	}
}

inline unsigned int GPU::get_act_sm() {
	return this->slots[0].get_sm() + this->slots[1].get_sm();
}

inline unsigned int GPU::get_rem_sm() {
	return this->total_sm - this->get_act_sm();
}


#endif // !__GPU_MODEL_H__