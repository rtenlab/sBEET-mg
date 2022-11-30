#ifndef __GPU_SLOT_MODEL_H__
#define __GPU_SLOT_MODEL_H__

#include <stdint.h>

#include "job_model.h"

class GPU_JobSlot {
public:
	GPU_JobSlot();
	~GPU_JobSlot();

	/* variables */
	Job *job; // no need to maintain multiple instance of a class. All the instances of jobs should be stored in `job_list` of class task, which this should point to.
	std::vector<Job> waitlist; // stores unreleased jobs, so not pointer here

	/* functions */
	inline const bool empty_() const;

	inline const unsigned int get_sm() const;
	inline void push_to_waitlist(Job _job); // don't use pointer
	inline void clear_waitlist();

	/* future status */
	inline const bool empty_tp(float tp) const; // equivalent of `get_slot_status_timepoint()
	const unsigned int get_sm_tp(float tp) const; // equivalent of `get_slot_sm_timepoint()`
	const app_t get_app_tp(float tp) const;

	friend class GPU;

protected:
	/* basics */
	void update_sm_map(uint64_t _sm_map);
	/* assign and delete */
	void assign_job_slot(Job *__job); // NOTE: in scheduler, the `job` must be declared as a pointer
	void delete_job_slot();
	/* power related */
	inline float calc_power_d(float *power_dynamic, Job *__job) const; // calculate the dynamic power of this slot
	
private:
	bool is_empty;
	unsigned int sm;
	uint64_t sm_map;
};

inline const bool GPU_JobSlot::empty_() const{
	return is_empty;
}

inline const bool GPU_JobSlot::empty_tp(float tp) const {
	if (!empty_()) {
		if (tp < this->job->gpu_end_time) {
			return false;
		}
	}
	for (auto it : this->waitlist) {
		if (tp < it.gpu_end_time && tp >= it.gpu_start_time) {
			return false;
		}
	}
	return true;
}

inline const unsigned int GPU_JobSlot::get_sm() const {
	return this->sm;
}

inline void GPU_JobSlot::push_to_waitlist(Job _job) {
	this->waitlist.push_back(_job);
}

inline void GPU_JobSlot::clear_waitlist() {
	this->waitlist.clear();
}

inline float GPU_JobSlot::calc_power_d(float *power_dynamic, Job *__job) const {
	if (__job == nullptr) {
		return 0;
	}
	return __job->get_sm() * power_dynamic[__job->app_id];
}


#endif // __GPU_SLOT_MODEL_H__
