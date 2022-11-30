#ifndef __WORKER_H__
#define __WORKER_H__

#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <iostream>
#include <queue>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <condition_variable>
#include <functional>

#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)

#include "job_model.h"
#include "common/include/messages.h"

struct worker_t {
	bool active;
	unsigned int task_id;
	unsigned int cpu_id;
	unsigned int gpu_id;
	cudaStream_t stream;
	int os_tid;
	pthread_t thread;
};


class WorkerPool {
public:
	WorkerPool (int thread_num, int gpu_id) : shutdown_ (false) {
		this->thread_num = thread_num;
		this->gpu_id = gpu_id;
		// cudaSetDevice(gpu_id);
		workers.reserve(thread_num);
		// threads.reserve(thread_num);
		for (int i = 0; i < thread_num; i++) {
			cudaStream_t stream;
			pthread_t new_thread;
			pthread_create(&new_thread, 0, &WorkerPool::threadEntryWrapper, this);
			std::string thread_name = "gpu_" + std::to_string(gpu_id) + "_thread_" + std::to_string(i);
			pthread_setname_np(new_thread, thread_name.c_str());
			worker_t temp = {false, 0, 8 + 4 * gpu_id + 2 * i, gpu_id, stream, -100, new_thread}; // bound to physical cores 8, 10, 12, 14
			workers.push_back(temp);
			// cudaStreamCreateWithFlags(&(workers[i].stream), cudaStreamNonBlocking);
		}

		for (int i = 0; i < thread_num; i++) {
			cudaStreamCreateWithFlags(&(workers[i].stream), cudaStreamNonBlocking);
		}


		pthread_mutex_init(&qmtx, NULL);
		pthread_cond_init(&wcond, NULL);
		// std::cout << "Creating threads..." << std::endl;
		// for (int i = 0; i < thread_num; ++i) {
		// 	pthread_t new_thread;
		// 	pthread_create(&new_thread, 0, &WorkerPool::threadEntryWrapper, this);
		// 	std::string thread_name = "gpu_" + std::to_string(gpu_id) + "_thread_" + std::to_string(i);
		// 	pthread_setname_np(new_thread, thread_name.c_str());
		// 	threads.push_back(new_thread);
		// }
	}

	~WorkerPool() {
		for (int i = 0; i < thread_num; i++) {
			workers[i].active = false;
			// std::cout << "Joining thread " << i << std::endl;
			// pthread_join(threads[i], NULL);
			pthread_join(workers[i].thread, NULL);
		}
		pthread_mutex_destroy(&qmtx);
		pthread_cond_destroy(&wcond);
	}

	std::vector<worker_t> workers;

	void recycle_worker(int id) {
		workers[id].active = false;
	}

	void stop() {
		pthread_mutex_lock(&qmtx);
		shutdown_ = true;
		pthread_cond_broadcast(&wcond);
		pthread_mutex_unlock(&qmtx);
	}

	void assign_worker(Job *job, int id) {
		if (id != 0 && id != 1) {
			id = find_avail_worker();
		}
		// updating job fields for execution
		// PRINT_VAL("Assigning worker ", id);
		job->cpu_id = workers[id].cpu_id;
		job->stream = &workers[id].stream;
		workers[id].active = true;
		workers[id].task_id = job->task_id;
		std::function<void(void)> func = std::bind(&Job::_jobExec, job);
		// Place a job on the queue and unblock a thread
		pthread_mutex_lock(&qmtx);
		target_tid = workers[id].os_tid;
		// std::cout << "waiting for thread " << target_tid << std::endl;
		jobs_.emplace(std::move(func));
		pthread_cond_signal(&wcond);
		pthread_mutex_unlock(&qmtx);
	}

	inline int find_avail_worker() {
		int id = -1;
		for (int i = 0; i < workers.size(); i++) {
			if (workers[i].active == false) {
				id = i;
				break;
			}
		}
		return id;
	}

protected:
	static void* threadEntryWrapper(void* param) {
		static_cast<WorkerPool*>(param)->threadEntry();
		return 0;
	}

	void threadEntry () {
		thread_local bool first_launch = true;
		thread_local int tid = gettid();
		while (1) {
			std::function<void(void)> foo;
			pthread_mutex_lock(&qmtx);
			if (target_tid != tid && !first_launch) {
				pthread_mutex_unlock(&qmtx);
				continue;
			}
			while (!shutdown_ && jobs_.empty()) {
				pthread_cond_wait(&wcond, &qmtx);
			}
			// std::cout << "Got a job: OS thread id = " << tid << std::endl;
			// NOTE: there is a 10ms overhead sometimes
			if (shutdown_) {
				return;
			}
			// std::cout << "thread " << tid << " responding" << std::endl;
			foo = std::move(jobs_.front());
			jobs_.pop();
			
			// if first launch, find a random worker to assign os_tid
			if (first_launch) {
				for (int i = 0; i < workers.size(); i++) {
					if (workers[i].os_tid == -100) {
						workers[i].os_tid = tid;
						// std::cout << "Update worker[" << i << "] os_tid: " << workers[i].os_tid << std::endl;
						cpu_set_t set;
						CPU_ZERO(&set);
						CPU_SET(workers[i].cpu_id, &set);
						sched_setaffinity(gettid(), sizeof(cpu_set_t), &set);
						PRINT_MSG("Thread " + std::to_string(tid) + " set to CPU " + std::to_string(workers[i].cpu_id) + ", bounded to GPU " + std::to_string(workers[i].gpu_id));
						break;
					}
				}
				first_launch = false;
			}
			
			pthread_mutex_unlock(&qmtx); // should not be here, scope yes
			// Do the job without holding any locks
			foo();
		}
	}

	int thread_num;
	unsigned int gpu_id;
	pthread_mutex_t qmtx;
	pthread_cond_t wcond;
	bool shutdown_;
	std::queue<std::function <void(void)>> jobs_;
	std::vector<pthread_t> threads;
	int target_tid;
};

#endif // !__WORKER_H__