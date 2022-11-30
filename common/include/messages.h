#ifndef __MESSAGES_H__
#define __MESSAGES_H__

#include <sys/time.h>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <string>
#include <sstream> 

/* messages */
// #define _VERBOSE
#define _RECORD_TO_FILE
// #define _POWER_TRACE

#define TIMESTAMP() { \
	auto now = std::chrono::system_clock::now(); \
	const auto now_tt = std::chrono::system_clock::to_time_t(now); \
	const auto now_micro = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()) % 1000000; \
	std::cout << std::put_time(std::localtime(&now_tt), "%m:%d:%Y %T") << "." << now_micro.count() << " -> "; \
}

#define TIMESTAMP_ARG(s) { \
	auto now = std::chrono::system_clock::now(); \
	const auto now_tt = std::chrono::system_clock::to_time_t(now); \
	const auto now_micro = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()) % 1000000; \
	std::stringstream ss; \
	ss << std::put_time(std::localtime(&now_tt), "%D %T") << "." << now_micro.count(); \
	s = ss.str(); \
}


#define PRINT_ELEMENT(width, ...) std::cout << std::left << std::setw(width) << std::setfill(' ') << __VA_ARGS__;

#ifdef _VERBOSE 
#define PRINT_TASKSET() { \
	PRINT_ELEMENT(12, "Task ID");	\
	PRINT_ELEMENT(12, "APP");	\
	PRINT_ELEMENT(12, "PERIOD");	\
	PRINT_ELEMENT(12, "GPU ID");	\
	PRINT_ELEMENT(12, "Mopt");	\
	PRINT_ELEMENT(12, "UTIL");	\
	PRINT_ELEMENT(12, "PHI");	\
	std::cout << std::endl;	\
	for (unsigned int i = 0; i < task_list.size(); i++) {	\
		PRINT_ELEMENT(12, i);	\
		PRINT_ELEMENT(12, task_list[i].app_id);	\
		PRINT_ELEMENT(12, task_list[i].period);	\
		PRINT_ELEMENT(12, task_list[i].gpu_id);	\
		PRINT_ELEMENT(12, task_list[i].mopt); \
		PRINT_ELEMENT(12, task_list[i].get_util(task_list[i].gpu_id));	\
		PRINT_ELEMENT(12, task_list[i].phi);	\
		std::cout << std::endl;	\
	}	\
}
#else
#define PRINT_TASKSET() do {} while (0);
#endif

#ifdef _VERBOSE
#define PRINT_MSG(...) TIMESTAMP(); std::cout << __VA_ARGS__ << std::endl;
#else
#define PRINT_MSG(...) do {} while(0);
#endif	

#ifdef _VERBOSE
#define PRINT_VAL(msg, val) TIMESTAMP(); std::cout << msg << ": " << val << std::endl;
#else
#define PRINT_VAL(workload, ...) do {} while(0);
#endif	

#ifdef _VERBOSE
#define WORKLOAD_INFO(workload, ...) TIMESTAMP(); std::cout << "[" << workload << "] " __VA_ARGS__ << std::endl;
#else
#define WORKLOAD_INFO(workload, ...) do {} while(0);
#endif	

#ifdef _VERBOSE
#define TASK_INFO(task_id, ...) TIMESTAMP(); std::cout << "[Task " << task_id << "] " __VA_ARGS__ << std::endl;
#else
#define TASK_INFO(task_id, ...) do {} while(0);
#endif	

#define PRINT_DUMMY(x) \
	for (int i = 0; i < 20; i++) std::cout << x; \
	std::cout << std::endl;



#endif // !__MESSAGES_H__