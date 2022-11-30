/* 
This file is not supposed to include any project header files
 */
#include <sys/time.h>
#include <iomanip>
#include <assert.h>
#include "support.h"

#include "common/include/defines.h"

namespace T400 {
	const int max_sm = 6;
	const int total_sm = sm_limit[1];
	const float power_static = 8;
	const float power_idle = 0.652;
	float power_dynamic[8] = {2.06, 0.98, 0.81, 1.15, 1.07, 1.29, 2.06, 1.19};
	float ghd[8] = {13, 1, 3, 12, 6, 4.5, 3, 2.5};
	float gdh[8] = {1.5, 1, 1.5, 4, 5.5, 1, 1.5, 1};
}

namespace RTX3070 {
	const int max_sm = 46;
	const int total_sm = sm_limit[0]; 
	const float power_static = 46;
	const float power_idle = 0.445;
	float power_dynamic[8] = {3.77, 1.63, 1.14, 1.67, 0.98, 1.08, 3.18, 0.91};
	float ghd[8] = {2.5, 4, 1.5, 1, 4.5, 2, 1.5, 1};
	float gdh[8] = {1, 1, 1, 1, 1, 1, 1, 1};
}

std::chrono::time_point<std::chrono::system_clock> timing::sys_start_time;

float timing::get_sys_time() {
	return (float)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sys_start_time).count()) / 1000;
}

void timing::recordStartTime(mytimer_t *mytimer) {
	mytimer->start_time = std::chrono::system_clock::now();
}

void timing::recordStopTime(mytimer_t *mytimer) {
	mytimer->stop_time = std::chrono::system_clock::now();
}

float timing::elapsedTime(mytimer_t mytimer) {
	return (float)(std::chrono::duration_cast<std::chrono::microseconds>(mytimer.stop_time - mytimer.start_time).count()) / 1000;
}

void timing::resetTimer(mytimer_t *mytimer) {
	mytimer->start_time = {};
	mytimer->stop_time = {};
}

// return the specified colummn of the csv
std::vector<float> read_csv_column(std::string fn, int cid) {
	std::vector<float> ret;
	std::fstream fin;
	fin.open(fn, std::ios::in);
	std::string line;

	std::string word;
	int row = 0;
	while (std::getline(fin, line)) {
		std::stringstream s(line);
		// std::cout << line << std::endl;
		row++;
		if (row == 1) 
			continue; // first row is title

		int index = 0;
		while (std::getline(s, word, ',')) {
			if (index == cid) {
				ret.push_back(std::stof(word));
			}
			index++;
		}
	}

	assert(!ret.empty());
	return ret;
}

unsigned int get_sm_by_map(uint64_t sm_map) {
	unsigned int cnt = 0;
	while (sm_map) {
		if (sm_map & 1)
			cnt++;
		sm_map = sm_map >> 1;
	}
	return cnt;
}
