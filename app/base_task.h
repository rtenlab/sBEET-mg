#ifndef __BASE_TASK_H__
#define __BASE_TASK_H__

#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

class BASE_TASK {
	public:
		BASE_TASK() {};
		~BASE_TASK() {};

		virtual void taskInit() { 
			std::cout << "taskInit() from base." << std::endl; 
		};
		virtual void taskInitDevice(unsigned int a) { 
			std::cout << "taskInitDevice() from base." << std::endl; 
		};
        virtual void taskPreGPU(cudaStream_t* s, unsigned int a) { 
			std::cout << "taskPreGPU() from base." << std::endl; 
		}; 
        virtual void taskPostGPU(unsigned int a) {
			std::cout << "taskPostGPU() from base." << std::endl;
		}; 
        virtual void taskRunOnGPU(unsigned int a, std::vector<int> &v) {
			std::cout << "taskRunOnGPU() from base." << std::endl;
		};
		virtual void taskFinish() {
			std::cout << "taskFinish() from base." << std::endl;
		};
		virtual void verify() {};
};


#endif // __BASE_TASK_H__