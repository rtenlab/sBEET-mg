#ifndef __SUPPORT_H__
#define __SUPPORT_H__

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>

namespace RTX3070 {
	extern const int max_sm;
	extern const int total_sm;
	extern const float power_static;
	extern const float power_idle;
	extern float power_dynamic[8];
	extern float ghd[8];
	extern float gdh[8];
}

namespace T400 {
	extern const int max_sm;
	extern const int total_sm;
	extern const float power_static;
	extern const float power_idle;
	extern float power_dynamic[8];
	extern float ghd[8];
	extern float gdh[8];
}

std::vector<float> read_csv_column(std::string fn, int cid);

unsigned int get_sm_by_map(uint64_t sm_map);

namespace timing {
	extern std::chrono::time_point<std::chrono::system_clock> sys_start_time;

	float get_sys_time();

	struct mytimer_t {
		std::chrono::time_point<std::chrono::system_clock> start_time;
		std::chrono::time_point<std::chrono::system_clock> stop_time;
	};

	void recordStartTime(mytimer_t*);
	void recordStopTime(mytimer_t*);
	float elapsedTime(mytimer_t);
	void resetTimer(mytimer_t*);
};

#endif // !__SUPPORT_H__