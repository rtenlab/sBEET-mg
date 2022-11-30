#include <sys/types.h>
#include <sys/fcntl.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

#include "bfs_kernel.cuh"
#include "bfs.h"
#include "common/include/STGM.h"
#include "common/include/messages.h"

// #define _VERBOSE

/* 
    Take the input arguments for this class
*/
BFS_TASK::BFS_TASK(char* filename) {
	fp = fopen(filename, "r");
	if(!fp) fprintf(stderr, "Error Reading graph file\n");

	fscanf(fp, "%d", &no_of_nodes);

    int num_of_blocks = 1;
    int num_of_threads_per_block = no_of_nodes;
    
    if(no_of_nodes > MAX_THREADS_PER_BLOCK) {
		num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
    }
    DimGrid = dim3(num_of_blocks, 1, 1);
	DimBlock = dim3(num_of_threads_per_block, 1, 1);

};

/*  */
BFS_TASK::~BFS_TASK() {};

/* 
    Init: 
    1, Malloc (host, device)
    2. definition of variables if necessary
*/
void BFS_TASK::taskInit() {
    // WORKLOAD_INFO("BFS", "Allocating variables...");
	h_graph_nodes = (Node*)malloc(sizeof(Node)*no_of_nodes);
	h_graph_mask = (bool*)malloc(sizeof(bool)*no_of_nodes);
	h_updating_graph_mask = (bool*)malloc(sizeof(bool)*no_of_nodes);
    h_graph_visited = (bool*)malloc(sizeof(bool)*no_of_nodes);
    
    int start, edgeno;   
    for( unsigned int i = 0; i < no_of_nodes; i++) {
		fscanf(fp, "%d %d", &start, &edgeno);
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i] = false;
		h_updating_graph_mask[i] = false;
		h_graph_visited[i] = false;
    }
    
	fscanf(fp, "%d", &source); //read the source node from the file
	source = 0;
	//set the source node as true in the mask
	h_graph_mask[source] = true;
    h_graph_visited[source] = true;
	
	fscanf(fp,"%d",&edge_list_size);
	
    int id, cost;
	h_graph_edges = (int*)malloc(sizeof(int)*edge_list_size);
	for(int i = 0; i < edge_list_size ; i++) {
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i] = id;
	}
    if(fp) fclose(fp);  
};

void BFS_TASK::taskInitDevice(unsigned int _id) {
  STGM_INIT(_id);
	cudaMalloc((void**) &plan[_id].d_graph_nodes, sizeof(Node)*no_of_nodes) ;
	cudaMalloc((void**) &plan[_id].d_graph_edges, sizeof(int)*edge_list_size) ;
	cudaMalloc((void**) &plan[_id].d_graph_mask, sizeof(bool)*no_of_nodes) ;
	cudaMalloc((void**) &plan[_id].d_updating_graph_mask, sizeof(bool)*no_of_nodes) ;
	cudaMalloc((void**) &plan[_id].d_graph_visited, sizeof(bool)*no_of_nodes) ;
	cudaMalloc((void**) &plan[_id].d_cost, sizeof(int)*no_of_nodes);
    cudaMalloc((void**) &plan[_id].d_over, sizeof(bool));

	// h_cost = (int*)malloc(sizeof(int)*no_of_nodes);
	cudaMallocHost((void**)&plan[_id].h_cost, sizeof(int)*no_of_nodes);
	for(int i = 0; i < no_of_nodes; i++) 
		plan[_id].h_cost[i]=-1;
	plan[_id].h_cost[source]=0;
    
    cudaDeviceSynchronize();
}

/* 
    Periodic CPU works PRE kernel launching
    1. fresh some values
    2. memcpy, async on a specific stream
*/
void BFS_TASK::taskPreGPU(cudaStream_t* s, unsigned int _id) {
	the_stream = s;
	cudaMemcpyAsync(plan[_id].d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice, *the_stream);
	cudaMemcpyAsync(plan[_id].d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice, *the_stream);
	cudaMemcpyAsync(plan[_id].d_graph_mask, h_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice, *the_stream);
	cudaMemcpyAsync(plan[_id].d_updating_graph_mask, h_updating_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice, *the_stream) ;
	cudaMemcpyAsync(plan[_id].d_graph_visited, h_graph_visited, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice, *the_stream);
    cudaMemcpyAsync(plan[_id].d_cost, plan[_id].h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice, *the_stream);
}

/*  */
void BFS_TASK::taskPostGPU(unsigned int _id) {
	// Timer mytimer;
    // startTime(&mytimer);
	cudaMemcpyAsync(plan[_id].h_cost, plan[_id].d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost, *the_stream);
	// stopTime(&mytimer);
	// printf("bfs memcpy d2h  time %f\n", elapsedTime(mytimer));
	
	// NOTE: saving results cost too much time
	// startTime(&mytimer);
	// FILE *fpo = fopen("result.txt","w");
	// for(int i = 0; i < no_of_nodes; i++)
	// 	fprintf(fpo,"%d) cost:%d\n", i, h_cost[i]);
	// fclose(fpo);
	// stopTime(&mytimer);
    // printf("bfs save result time %f\n", elapsedTime(mytimer));
}

/* 
    Launch kernel:
    1. SM allocation, decided by the algorithm
    2. launch the kernel(s)
*/
void BFS_TASK::taskRunOnGPU(unsigned int _id, std::vector<int> &_sm_arr) {
	// int sm_num = _sm_arr.size();
	int __n = _sm_arr.size();
    for (int i = 0; i < plan[_id].sm_num - __n; i++) {
        _sm_arr.push_back(-1);
    }
    STGM_SM_MAPPING(plan[_id].sm_num, _sm_arr);

	int k = 0;
	bool stop = 0;

	do {
		stop = false;
		cudaMemcpyAsync(plan[_id].d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice, *the_stream);
		STGM_LAUNCH_KERNEL(_id, Kernel1, *the_stream, DimGrid, DimBlock, plan[_id].d_graph_nodes, plan[_id].d_graph_edges, plan[_id].d_graph_mask, plan[_id].d_updating_graph_mask, plan[_id].d_graph_visited, plan[_id].d_cost, no_of_nodes);
		STGM_LAUNCH_KERNEL(_id, Kernel2, *the_stream, DimGrid, DimBlock, plan[_id].d_graph_mask, plan[_id].d_updating_graph_mask, plan[_id].d_graph_visited, plan[_id].d_over, no_of_nodes);
		cudaMemcpyAsync(&stop, plan[_id].d_over, sizeof(bool), cudaMemcpyDeviceToHost, *the_stream);
		k++;
	//} while(stop);
	} while(k < 100); // Enforce fixed number of iterations
}

/*  */
void BFS_TASK::taskFinish() {
	free(h_graph_nodes);
	free(h_graph_edges);
	free(h_graph_mask);
	free(h_updating_graph_mask);
	free(h_graph_visited);
	
	for (int _id = 0; _id < 2; _id++) {
		cudaFreeHost(plan[_id].h_cost);
		cudaFree(plan[_id].d_graph_nodes);
		cudaFree(plan[_id].d_graph_edges);
		cudaFree(plan[_id].d_graph_mask);
		cudaFree(plan[_id].d_updating_graph_mask);
		cudaFree(plan[_id].d_graph_visited);
		cudaFree(plan[_id].d_cost);
		STGM_FINISH(_id);
	}
}
