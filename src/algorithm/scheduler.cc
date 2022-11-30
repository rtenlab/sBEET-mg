#include <algorithm>
#include <math.h>

#include "scheduler.h"
#include "src/model/support.h"
#include "src/model/worker.h"
#include "src/model/job_model.h"
#include "src/model/power_model.h"
#include "common/include/defines.h"
#include "common/include/messages.h"

#include "app/mmul/mmul.h"
#include "app/stereodisparity/stereodisparity.h"
#include "app/hotspot/hotspot.h"
#include "app/dxtc/dxtc.h"
#include "app/bfs/bfs.h"
#include "app/histogram/histogram.h"

using namespace runtime;
using namespace timing;
using namespace rt_task;

std::vector<Task> runtime::task_list;
std::vector<GPU> runtime::gpu_list;

float runtime::energy_pred;
std::string runtime::filename;
std::string runtime::policy_name;
std::string runtime::sched_start_time;
std::string runtime::sched_end_time;
float runtime::duration;

/* Job management */
std::vector<Job*> runtime::ready_queue;
std::unordered_set<unsigned int> runtime::running_tasks;
std::vector<Job*> runtime::yield_queue;
std::vector<Job*> runtime::job_latest;
int runtime::released = 0;
int runtime::missed = 0;
int runtime::completed = 0;

bool task_util_comparator(const Task & a, const Task & b) {
	return a.ref_util > b.ref_util;
}

bool task_period_comparator(const Task & a, const Task & b) {
	return a.period < b.period;
}

bool job_period_comparator_ptr(const Job * a, const Job * b) {
	return a->task->period < b->task->period;
}

void sort_ready_queue(std::vector<Job*> & _ready_queue, std::string _policy) {
	if (_policy == "FIFO") {}
	else if (_policy == "RM") {
		std::sort(_ready_queue.begin(), _ready_queue.end(), job_period_comparator_ptr);
	}
}

/* Once a job of a task is released, the previous job will be removed */
void runtime::release() {
	float period_prev;
	for (int i = 0; i < task_list.size(); i++) {
		if (job_latest[i] == nullptr) {
			period_prev = task_list[i].phi;
		}else { // in case period != deadline
			period_prev = job_latest[i]->release_time + job_latest[i]->task->period;
		}

		if (get_sys_time() >= period_prev) {
			// generate a job
			job_latest[i] = new Job(&task_list[i], task_list[i].gpu_id, get_sys_time(), get_sys_time() + task_list[i].period * deadline_factor);
			// task.jobs.push_back(job);
			ready_queue.push_back(job_latest[i]);
			TASK_INFO(job_latest[i]->task_id, "is released.");
			released += 1;
		}
	}
	sort_ready_queue(ready_queue, "RM");
}

void runtime::complete() {
	// by checking each `rt_status`
	int rid = 0;
	for (int i = 0; i < gpu_list.size(); i++) {
		for (int j = 0; j < partition_num; j++) {
			shared_mutex.lock();
			rid = i * partition_num + j;
			if (rt_vec[rid].valid == true && rt_vec[rid].status == rt_task::FREED | rt_task::GPU_DONE) { 
				// reset the slot no matter what is the rt state
				if (gpu_list[i].slots[j].job == nullptr) {
					continue;
				}else if (gpu_list[i].slots[j].job->task_id == rt_vec[rid].task_id) {
					if (rt_vec[rid].status == rt_task::GPU_DONE)
						yield_queue.push_back(gpu_list[i].slots[j].job);
					gpu_list[i].reset_slot(j);
				}
				// recycle the worker if the job is completed
				if (rt_vec[rid].status == rt_task::FREED) { // extra steps if a job is fully completed
					if (gpu_list[i].worker_pool->workers[j].task_id == rt_vec[rid].task_id) {
						gpu_list[i].worker_pool->recycle_worker(j);
					}
				}
			}
			// check whether deadline missed, (`valid` is true)
			if (rt_vec[rid].valid == true && (rt_vec[rid].completed == false || get_sys_time() > job_latest[rt_vec[rid].task_id]->deadline)) {
				missed++;
				TASK_INFO(rt_vec[rid].task_id, "Missed");
				running_tasks.erase(rt_vec[rid].task_id);
			}else if (rt_vec[rid].valid == true && rt_vec[rid].completed == true) {
				TASK_INFO(rt_vec[rid].task_id, "Completed");
				running_tasks.erase(rt_vec[rid].task_id);
				// PRINT_MSG("GPU " + std::to_string(gpu_list[i].gpu_id) + " after complete status: " + std::to_string(gpu_list[i].get_status()));
				completed++;
			}
			rt_vec[rid].valid = false;
			shared_mutex.unlock();
		}
	}
}

/* skip the missed unreleased job, remove from the ready queue */
// it is ok to remove them from `ready_queue`, since there is still a copy as `job_latest`
void runtime::skip() {
	std::vector<int> rev_index;
	for (int i = 0; i < ready_queue.size(); i++) {
		if (get_sys_time() >= ready_queue[i]->deadline) {
			rev_index.push_back(i);
			missed++;
			TASK_INFO(ready_queue[i]->task_id, "Skipped. ");
		}
	}

	int cnt = 0;
	for (auto & it : rev_index) {
		ready_queue.erase(ready_queue.begin() + (it + cnt));
		cnt--;
	}
}

void runtime::execute(Job *job) {
	// PRINT_MSG("GPU " + std::to_string(job->gpu_id) + " before excute status: " + std::to_string(gpu_list[job->gpu_id].get_status()));
	job->update_config(job->gpu_id, job->get_sm_map(), get_sys_time(), 1);
	unsigned int sid = gpu_list[job->gpu_id].assign_slot(job, -1);
	gpu_list[job->gpu_id].reset_slot_waitlist(sid);
	TASK_INFO(job->task_id, "Start execution on GPU " + std::to_string(job->gpu_id) + " SM " + std::to_string(job->get_sm()) + "(" + std::to_string(gpu_list[job->gpu_id].get_rem_sm()) + ")");
	gpu_list[job->gpu_id].worker_pool->assign_worker(job, -1); // non-specified worker id
	// PRINT_MSG("GPU " + std::to_string(job->gpu_id) + " after excute status: " + std::to_string(gpu_list[job->gpu_id].get_status()));
	// put the task id into running_tasks
	running_tasks.insert(job->task->task_id);
	// std::cout << "Running tasks: {";
	// for (const auto& elem: running_tasks) {
	// 	std::cout << elem << ",";
	// }
	// std::cout << "}" << std::endl;

}

/* load taskset to `task_list` from a csv file */
void runtime::loadTaskset() {
	std::cout << filename << std::endl;
	std::vector<float> app_list = read_csv_column(filename, 0);
	std::vector<float> period_list = read_csv_column(filename, 1);
	std::vector<float> phi_list = read_csv_column(filename, 2);
	std::vector<float> util_list = read_csv_column(filename, 3);

	for (int i = 0; i < app_list.size(); i++) {
		// task_list.push_back(Task(app_t(app_list[i]), period_list[i], phi_list[i], util_list[i]));
		task_list.push_back(Task(app_t(app_list[i]), period_list[i], phi_list[i], util_list[i]));
	}

	/* 
	 * As long as nothing to do with `task_id`, all the init stuff can be done here
	 */
	for (int i = 0; i < task_list.size(); i++) {
		task_list[i].init_gpu_rankings();
		job_latest.push_back(nullptr);
	}
}

void runtime::initGPUs() { // TODO: later put this into a json config file
	gpu_list.push_back(GPU(0, "rtx3070"));
	gpu_list.push_back(GPU(1, "t400"));
}

inline BASE_TASK* runtime::createWorkload(app_t _app) {
	BASE_TASK *_workload;
	if (_app == MMUL) {_workload = new MMUL_TASK(2048, 1024, 2048);}
	// if (_app == MMUL) {_workload = new MMUL_TASK(8, 8, 8);}
	else if (_app == STEREODISPARITY) {_workload = new STEREODISPARITY_TASK();}
	else if (_app == HOTSPOT) {_workload = new HOTSPOT_TASK(1024, 2, 50, "app/hotspot/data/temp_1024", "app/hotspot/data/power_1024", "app/hotspot/output.out");}
	else if (_app == DXTC) {_workload = new DXTC_TASK("app/dxtc/data/lena-orig.ppm");}
	else if (_app == BFS) {_workload = new BFS_TASK("app/bfs/data/graph1MW_6.txt");}
	else if (_app == HIST) {_workload = new HIST_TASK(3200000, 4096);}
	else if (_app == MMUL2) {_workload = new MMUL_TASK(1024, 1024, 1024);}
	else if (_app == HIST2) {_workload = new HIST_TASK(1600000, 4096);}
	return _workload;
}

unsigned int runtime::find_gpu_with_min_util() {
	float val = std::numeric_limits<float>::max();
	unsigned int index = 0;
	for (unsigned int i = 0; i < gpu_list.size(); i++) {
		if (gpu_list[i].util < val) {
			val = gpu_list[i].util;
			index = i;
		}
	}
	return index;
}

/* 
 * find the gpu with lower util after the task is aloocated 
 * Like Load-Dist
*/
// TODO: for mg, use mopt to compute the remaining space of each GPU
unsigned int runtime::find_gpu_with_min_cap(Task &task) {
	float* temp_util = new float[gpu_list.size()];
	for (int i = 0; i < gpu_list.size(); i++) { 
		temp_util[i] = gpu_list[i].util + task.get_util(gpu_list[i].gpu_id);
	}
	float val = std::numeric_limits<float>::max();
	int index = -1;
	for (int i = 0; i < gpu_list.size(); i++) {
		if (temp_util[i] < val) {
			val = temp_util[i];
			index = i;
		}
	} 
	return gpu_list[index].gpu_id;
}

unsigned int runtime::find_energy_optimal_gpu(Task &task) {
	// std::cout << task.task_id << " assigned to energy opt gpu " << task.gpu_rankings[0].gpu_id << std::endl;
	return task.gpu_rankings[0].gpu_id;
}


void runtime::Init(std::string fn) {
	task_list.reserve(10);
	gpu_list.reserve(2);
	ready_queue.reserve(20);
	job_latest.reserve(20);
	yield_queue.reserve(20);
	rt_vec.reserve(4);

	filename = fn;

	PRINT_MSG("Initializing GPUs.");
	initGPUs();
	
	PRINT_MSG("Loading taskset.");
	loadTaskset();

	PRINT_MSG("Distributing tasks.");
	if (policy_name == "mg-jm" || policy_name == "mg-offline" || policy_name == "mg-sbeet") {
		mg::taskDistribution();
		PRINT_TASKSET();
	}else if (policy_name == "lcf" || policy_name == "bcf" || policy_name == "ld") {
		base_online::taskDistribution();
	}else if (policy_name == "bfd-sbeet" || policy_name == "ffd-sbeet" || policy_name == "wfd-sbeet") {
		base_offline::taskDistribution();
		PRINT_TASKSET();
	}else {
		std::cout << "Unknown policy name. Abort!" << std::endl;
		exit(1);
	}

	energy_pred = 0;

	for (int i = 0; i < partition_num * gpu_list.size(); i++) {
		rt_vec.push_back({rt_task::READY, 100, false, false, false}); // status, task_id, valid, completed, shutdown
	}

	for (auto & task : task_list) {
		task.workload = createWorkload(task.app_id);
		task._taskInit();
	}
	
	for (int i = 0; i < gpu_list.size(); i++) {
		cudaSetDevice(i);
		for (auto & task : task_list) {
			task._taskInitDevice(i);
			// task._taskInit(0);
		}
	}
	// std::cout << "---------------------------------------------" << std::endl;
}

/* Execute each workload once before scheduler starts */
void runtime::preLaunch() {
	for (int j = 0; j < gpu_list.size(); j++) {
		for (int i = 0; i < task_list.size(); i++) {
			for (int wid = 0; wid < partition_num; wid++) {
				Job * job = new Job(&task_list[i], gpu_list[j].gpu_id, get_sys_time(), get_sys_time()+ task_list[i].period);
				job->update_config(gpu_list[j].gpu_id, std::pow(2, sm_limit[j]) - 1, get_sys_time(), 1);
				job->first_launch = true;
				unsigned int sid = gpu_list[j].assign_slot(job, wid);
				gpu_list[j].worker_pool->assign_worker(job, wid);

				// wait for it to come back
				bool flag = false;
				while (!flag) {
					int rid = 0;
					for (int k = 0; k < gpu_list.size(); k++) {
						for (int p = 0; p < partition_num; p++) {
							shared_mutex.lock();
							rid = k * partition_num + p;
							// std::cout << "Gpu " << k << " slot " << j << std::endl;
							if (rt_vec[rid].valid == true && rt_vec[rid].status == rt_task::FREED) {
								if (gpu_list[k].slots[p].job == nullptr) {
									shared_mutex.unlock();
									continue;
								}else {
									gpu_list[k].reset_slot(p);
									gpu_list[k].worker_pool->recycle_worker(p);
									rt_vec[rid].status = rt_task::READY;
									rt_vec[rid].valid = false;
									flag = true;
								}
							}
							shared_mutex.unlock();
						}
					}
				}
			} // wid loop ends
		} // task loop ends
	} // gpu loop ends
}

void runtime::shutDown() {
	PRINT_MSG("System shutting down...");
	for (auto & rt : rt_vec) {
		shared_mutex.lock();
		rt.shutdown = true;
		shared_mutex.unlock();
	}

	for (int i = 0; i < gpu_list.size(); i++) {
		gpu_list[i].worker_pool->stop();
	}

	for (auto & task : task_list) {
		task._taskComplete();
	}

	int total_jobs = 0;
	for (auto task : task_list) {
		total_jobs += duration / task.period + 1;
	}
	// print summary
	PRINT_VAL("Total number of released", released);
	// PRINT_VAL("Total number of missed", missed);
	PRINT_VAL("Total number of missed", released - completed);
	PRINT_VAL("Total predicted energy", energy_pred);
#ifdef _RECORD_TO_FILE
	// create a csv file and write the summary, w/o title
	// title: filename(full), sched_start_time, sched_end_time, released, missed, energy_pred
	std::size_t pos1 = filename.find_first_of("/") + 1;
	std::size_t pos2 = filename.find("/set_u");
	// std::string fn = "output/" + filename.substr(pos1, 16) + "/" + policy_name + filename.substr(pos2, 8) + ".csv";
	std::string fn = "output/taskset_08022022/" + policy_name + "/set_u" + filename.substr(pos2 + 6, 2) + ".csv";
	// std::cout << fn << std::endl;
	std::fstream fout;
	fout.open(fn, std::ios::out | std::ios::app);
	fout << filename << "," << sched_start_time << "," << sched_end_time << "," << released << "," << missed << "," << energy_pred << std::endl;
#endif // _RECORD_TO_FILE
}

void runtime::Run(std::string filename, float _duration, std::string _policy_name) {
	policy_name = _policy_name;
	duration = _duration;
	/* Init */
	Init(filename);

	if (policy_name == "mg-jm" || policy_name == "mg-sbeet") {
		mg::schedulerOnline();
	}else if (policy_name == "lcf" || policy_name == "bcf" || policy_name == "ld") {
		base_online::schedulerOnline();
	}else if (policy_name == "mg-offline") {
		base_online::schedulerOnline();
	}else if (policy_name == "bfd-sbeet" || policy_name == "ffd-sbeet" || policy_name == "wfd-sbeet") {
		mg::schedulerOnline();
	}

	shutDown();
}

/* 
 * functions in namespace mg
 */
void runtime::mg::taskDistribution() {
	//  the input list of tasks must be sorted with priority order
	std::sort(task_list.begin(), task_list.end(), task_period_comparator);
	for (auto & task : task_list) {
		bool isAssigned = false;
		for (auto & tg : task.gpu_rankings) {
			if (gpu_list[tg.gpu_id].util + task.get_util(tg.gpu_id) <= 1) {
				task.gpu_id = tg.gpu_id;
				task.mopt = tg.mopt;
				gpu_list[tg.gpu_id].util += task.get_util(tg.gpu_id);
				isAssigned = true;
				break;
			}
		}
		if (isAssigned == false) {
			task.gpu_id = find_gpu_with_min_cap(task);
			unsigned int idx = task.search_in_gpu_rankings(task.gpu_id);
			task.mopt = task.gpu_rankings[idx].mopt;
			gpu_list[task.gpu_id].util += task.get_util_mopt(task.gpu_id);
			isAssigned = true;
		}
	}
	std::sort(task_list.begin(), task_list.end(), task_period_comparator);
	for (unsigned int i = 0; i < task_list.size(); i++) {
		task_list[i].task_id = i;
	}
}

void runtime::mg::schedulerOnline() {
	for (int i = 0; i < gpu_list.size(); i++) {
		cudaSetDevice(i);
		gpu_list[i].worker_pool = new WorkerPool(partition_num, i);
	}
	sys_start_time = std::chrono::system_clock::now();
	
	PRINT_MSG("Prelaunching the kernels...");
	preLaunch();
	
	sys_start_time = std::chrono::system_clock::now();
	PRINT_MSG("Scheduler starts.");
	TIMESTAMP_ARG(sched_start_time);
	int time_prev = -1;
	while (get_sys_time() < duration) {
		float loop_start_time = get_sys_time();

		/* scheduler starts */
		// check completion
		// complete_gpu();
		complete();
		skip();
		release();

		std::vector<int> rev_index;
		for (int i = 0; i < ready_queue.size(); i++) { // break until every job is visited or all the GPU is full
			if (running_tasks.find(ready_queue[i]->task->task_id) != running_tasks.end()) {
				// the one instance of the task is running
				continue;
			}
			bool no_rsrc = true;
			for (auto gpu : gpu_list) {
				if (gpu.get_status() != FULL)
					no_rsrc = false;
			}
			if (no_rsrc == true) {
				break;
			}
			Job *job;
			if (policy_name == "mg-jm") {
				job = job_migration(task_list, gpu_list, ready_queue[i]);
			}else { // including mg-sbeet, wfd-sbeet, ffd-sbeet, bfd-sbeet
				job = sBeetWrapper(task_list, gpu_list, &gpu_list[ready_queue[i]->task->gpu_id], ready_queue[i]);
			}
			if (job != nullptr) {
				*ready_queue[i] = *job;
				execute(ready_queue[i]);
				rev_index.push_back(i);
			}
		}
		int cnt = 0;
		for (auto & it : rev_index) {
			ready_queue.erase(ready_queue.begin() + (it + cnt));
			cnt--;
		}
		
		while (get_sys_time() - loop_start_time <= 0.1);
		if (int(get_sys_time()) != time_prev) {
			time_prev = int(get_sys_time());
			float power_instant = Power::compute_tick_energy(gpu_list);
			energy_pred += power_instant;
#ifdef _POWER_TRACE
			std::size_t pos2 = filename.find("/set_u");
			std::string fn = "output/taskset_08022022/" + policy_name + "/power_pred_" + filename.substr(pos2 + 6, 2) + ".csv";
			std::fstream fout;
			fout.open(fn, std::ios::out | std::ios::app);
			std::string t;
			TIMESTAMP_ARG(t);
			fout << t << "," << power_instant << std::endl;
#endif
		}
	}
	PRINT_MSG("Scheduler finishes.");
	TIMESTAMP_ARG(sched_end_time);
}

/* 
 * Generic task distribution for online baselines
 * 1. Sort with priority order and assign task_id
 * 2. mopt should be found at runtime, according to the available resources
 */
void runtime::base_online::taskDistribution() {
	std::sort(task_list.begin(), task_list.end(), task_period_comparator);
	for (unsigned int i = 0; i < task_list.size(); i++) {
		task_list[i].task_id = i;		
	}
}

Job* runtime::base_online::decisionHelper(Job* job) {
	if (policy_name == "lcf") {
		for (int i = gpu_list.size() - 1; i >= 0; i--) {
			if (gpu_list[i].get_status() != FULL) {
				// find max(mopt, avail)
				unsigned int m = MIN(job->task->get_mopt(gpu_list[i].gpu_id), gpu_list[i].get_rem_sm());
				job->update_config(gpu_list[i].gpu_id, gpu_list[i].gen_sm_map(m), get_sys_time(), 0);
				return job;
			}
		}
		return nullptr; // no available resources
	}else if (policy_name == "bcf") {
		for (int i = 0; i < gpu_list.size(); i++) {
			if (gpu_list[i].get_status() != FULL) {
				// find max(mopt, avail)
				unsigned int m = MIN(job->task->get_mopt(gpu_list[i].gpu_id), gpu_list[i].get_rem_sm());
				job->update_config(gpu_list[i].gpu_id, gpu_list[i].gen_sm_map(m), get_sys_time(), 0);
				return job;
			}
		}
		return nullptr; // no available resources
	}else if (policy_name == "ld") {
		for (int i = 0; i < gpu_list.size(); i++) { // first check if any GPU is empty
			if (gpu_list[i].get_status() == IDLE) {
				unsigned int m = MIN(job->task->get_mopt(gpu_list[i].gpu_id), gpu_list[i].get_rem_sm());
				job->update_config(gpu_list[i].gpu_id, gpu_list[i].gen_sm_map(m), get_sys_time(), 0);
				return job;
			}
		}
		// Neither of the GPU is IDLE
		unsigned int val = 0;
		unsigned int index = 0;
		for (int i = 0; i < gpu_list.size(); i++) { // if not assigned, assign to the GPU with more remaining SMs
			if (gpu_list[i].get_status() != FULL && gpu_list[i].get_rem_sm() > val) {
				val = gpu_list[i].get_rem_sm();
				index = i;
			}
		}
		if (val > 0 && index >= 0) { // there is at least one available GPU
			unsigned int m = MIN(job->task->get_mopt(gpu_list[index].gpu_id), gpu_list[index].get_rem_sm());
			job->update_config(gpu_list[index].gpu_id, gpu_list[index].gen_sm_map(m), get_sys_time(), 0);
			return job;
		}else {
			return nullptr; // no available resources
		}
	}else if (policy_name == "mg-offline") { // NOTE: test scenario when online alg of mg is not involved
		if (gpu_list[job->task->gpu_id].get_status() != FULL && gpu_list[job->task->gpu_id].get_rem_sm() >= job->task->mopt) {
			job->update_config(job->task->gpu_id, gpu_list[job->task->gpu_id].gen_sm_map(job->task->mopt), get_sys_time(), 0);
			return job;
		}else {
			return nullptr;
		}
	}
}

void runtime::base_online::schedulerOnline() {
	for (int i = 0; i < gpu_list.size(); i++) {
		cudaSetDevice(i);
		gpu_list[i].worker_pool = new WorkerPool(partition_num, i);
	}
	sys_start_time = std::chrono::system_clock::now();
	
	PRINT_MSG("Prelaunching the kernels...");
	preLaunch();

	sys_start_time = std::chrono::system_clock::now();
	PRINT_MSG("Scheduler starts.");
	TIMESTAMP_ARG(sched_start_time);
	int time_prev = -1;
	while (get_sys_time() < duration) {
		float loop_start_time = get_sys_time();

		/* scheduler starts */
		// check completion
		complete();
		skip();
		release();

		std::vector<int> rev_index;
		for (int i = 0; i < ready_queue.size(); i++) {
			if (running_tasks.find(ready_queue[i]->task->task_id) != running_tasks.end()) {
				// the one instance of the task is running
				continue;
			}
			bool no_rsrc = true;
			for (auto gpu : gpu_list) {
				if (gpu.get_status() != FULL)
					no_rsrc = false;
			}
			if (no_rsrc == true) {
				break;
			}
			Job *job = decisionHelper(ready_queue[i]);
			if (job != nullptr) {
				*ready_queue[i] = *job;
				execute(ready_queue[i]);
				rev_index.push_back(i);
			}
		}
		int cnt = 0;
		for (auto & it : rev_index) {
			ready_queue.erase(ready_queue.begin() + (it + cnt));
			cnt--;
		}
		while (get_sys_time() - loop_start_time <= 0.1);
		if (int(get_sys_time()) != time_prev) {
			time_prev = int(get_sys_time());
			energy_pred += Power::compute_tick_energy(gpu_list);
		}
	}
	PRINT_MSG("Scheduler finishes.");
	TIMESTAMP_ARG(sched_end_time);
}

struct bin_t {
	float util;
	unsigned int gpu_id;
};

bool bin_util_decreasing(const bin_t &a, const bin_t &b) {
	return a.util > b.util;
}

bool bin_util_increasing(const bin_t &a, const bin_t &b) {
	return a.util < b.util;
}

/* 
 * Simple offline task assignation methods
 * Assuming the "bins" to be identical
 */
void runtime::base_offline::taskDistribution() {
	std::sort(task_list.begin(), task_list.end(), task_util_comparator); // Largest-Task-First
	std::vector<bin_t> bins;
	for (auto g : gpu_list) {
		bins.push_back({0, g.gpu_id});
	}
	for (auto & task : task_list) {
		if (policy_name == "wfd-sbeet") {
			std::sort(bins.begin(), bins.end(), bin_util_increasing); // sort bins in increasing order of util
			// fit into the first bin, add to the util
			task.gpu_id = bins[0].gpu_id;
			unsigned int idx = task.search_in_gpu_rankings(task.gpu_id);
			task.mopt = task.gpu_rankings[idx].mopt;
			bins[0].util += task.ref_util;
		}else if (policy_name == "bfd-sbeet") {
			std::sort(bins.begin(), bins.end(), bin_util_decreasing); // sort bins in decreasing order of util
			for (auto & bin : bins) {
				if (bin.util + task.ref_util <= 1.0) { // fixme: this should be guaranteed that all the tasks can be assigned
					task.gpu_id = bin.gpu_id;
					unsigned int idx = task.search_in_gpu_rankings(task.gpu_id);
					task.mopt = task.gpu_rankings[idx].mopt;
					bin.util += task.ref_util;
					break;
				}
			}
		}else if (policy_name == "ffd-sbeet") {
			for (auto & bin : bins) {
				if (bin.util + task.ref_util <= 1.0) { // fixme: this should be guaranteed that all the tasks can be assigned
					task.gpu_id = bin.gpu_id;
					unsigned int idx = task.search_in_gpu_rankings(task.gpu_id);
					task.mopt = task.gpu_rankings[idx].mopt;
					bin.util += task.ref_util;
					break;
				}
			}
		}
	}
	std::sort(task_list.begin(), task_list.end(), task_period_comparator);
	for (unsigned int i = 0; i < task_list.size(); i++) {
		task_list[i].task_id = i;
	}
}