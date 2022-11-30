#ifndef __SCHEDULER_H__
#define __SCHEDULER_H__

#include <vector>
#include <string>
#include <unordered_set>
#include "src/model/support.h"
#include "src/model/task_model.h"
#include "src/model/job_model.h"
#include "src/model/gpu_model.h"

extern Job* job_migration(std::vector<Task> & task_list, std::vector<GPU> & gpu_list,Job *job);
extern Job* sBeetWrapper(std::vector<Task> & task_list, std::vector<GPU> & gpu_list, GPU* gpu, Job *job);

namespace runtime {
	/* global variables */
	extern std::vector<Task> task_list;
	extern std::vector<GPU> gpu_list;

	extern float energy_pred; // the predicted total energy of the system
	extern std::string filename;
	extern std::string policy_name;
	extern std::string sched_start_time;
	extern std::string sched_end_time;
	extern float duration;

	/* global functions */
	void Init(std::string fn);
	void shutDown();
	void Run(std::string filename, float duration, std::string policy_name);
	
	/* local functions */
	void loadTaskset();
	void initGPUs();	
	inline BASE_TASK* createWorkload(app_t _app);
	void preLaunch();

	unsigned int find_gpu_with_min_util();
	unsigned int find_gpu_with_min_cap(Task &task);
	unsigned int find_energy_optimal_gpu(Task &task);

	namespace mg {
		void taskDistribution(); 
		void schedulerOnline();
	}

	/* baselines of online algorithms: Loadt-Dist. LCF, BCF  */
	namespace base_online {
		void taskDistribution();
		Job* decisionHelper(Job*);
		void schedulerOnline();
	}

	/* baselines of offline algorithms: WFD, BFD, FFD */
	namespace base_offline {
		void taskDistribution(); // depending on policy name
	}

	/* job management */
	void release();
	void complete_gpu();
	void complete();
	void execute(Job *job);
	void skip();

	/* variables */
	extern std::vector<Job*> ready_queue;
	extern std::unordered_set<unsigned int> running_tasks;
	extern std::vector<Job*> yield_queue;	// stores the job that already yields the GPU but still in execution
	extern std::vector<Job*> job_latest;	// stores the latest job for each task
	extern int released;
	extern int missed;
	extern int completed;
}

#endif // !__SCHEDULER_H__