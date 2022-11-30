#include <string>
#include <unistd.h>
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)

#include "src/algorithm/scheduler.h"

/* NOTE: `duration` should be big enough if `e_mode` is ON */
int main(int argc, char **argv) {
	int opt;
	std::string filename;
	std::string policy_name;
	float duration = 0.0f; // seconds
	while ((opt = getopt(argc, argv, "hf:d:p:")) != EOF) {
		switch (opt) {
			case 'h':
				std::cout << "Usage: " << std::endl;
				std::cout << "	-h Display usages and help" << std::endl;
				std::cout << "	-f Path to the taskset" << std::endl;
				std::cout << "	-d Emulating duration (s)" << std::endl;
				std::cout << "	-p Scheduling policies: \n"
											<< "		sbeet-mg: the proposed work in RTSS 2022 publication \n"
											<< "		lcf: little-core-first method \n"
											<< "		bcf: big-core-first method \n"
											<< "		ld: load distribution method \n"
											<< "		bfd-sbeet: best-fit-decreasing offline heuristic with original sBEET \n"
											<< "		ffd-sbeet: first-fit-decreasing offline heuristic with original sBEET \n"
											<< "		wfd-sbeet: worst-fit-decreasing offline heuristic with original sBEET \n";
				exit(0);
			case 'f':
				filename.assign(optarg);
				break;
			case 'd':
				duration = atof(optarg);
				break;
			case 'p':
				policy_name.assign(optarg);
				break;
			case '?':
				std::cerr << "Invalid option: " << opt << std::endl;
			default:
				std::cout << std::endl;
				abort();
		}
	}

	/* Let it run on the first 2 cpu cores */
	cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(6, &set);
    sched_setaffinity(gettid(), sizeof(cpu_set_t), &set);

	runtime::Run(filename, duration * 1000, policy_name);
	
	return 0;
}