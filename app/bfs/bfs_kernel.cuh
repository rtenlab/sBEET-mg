#ifndef __BFS_KERNEL_H__
#define __BFS_KERNEL_H__

#include "common/include/STGM.h"
#include "bfs.h"

#define MAX_THREADS_PER_BLOCK 512

STGM_DEFINE_KERNEL(Kernel1, Node* g_graph_nodes, int* g_graph_edges, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited, int* g_cost, int no_of_nodes)
{
	KERNEL_PROLOGUE();
	KERNEL_PROLOGUE_2();
	int tid = BLOCKIDX_X*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if( tid<no_of_nodes && g_graph_mask[tid]) {
		g_graph_mask[tid]=false;
		for(int i=g_graph_nodes[tid].starting; i<(g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++) {
			int id = g_graph_edges[i];
			if(!g_graph_visited[id]) {
                g_cost[id]=g_cost[tid]+1;
				// printf("id, tid, %d, %d\n", id, tid);
                g_updating_graph_mask[id]=true;
            }
        }
	}
	KERNEL_EPILOGUE();
}

STGM_DEFINE_KERNEL(Kernel2, bool* g_graph_mask, bool *g_updating_graph_mask, bool* g_graph_visited, bool *g_over, int no_of_nodes)
{
	KERNEL_PROLOGUE();
	KERNEL_PROLOGUE_2();
	int tid = BLOCKIDX_X*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if( tid<no_of_nodes && g_updating_graph_mask[tid])
	{

		g_graph_mask[tid]=true;
		g_graph_visited[tid]=true;
		*g_over=true;
		g_updating_graph_mask[tid]=false;
	}
	KERNEL_EPILOGUE();
}

#endif /* __BFS_KERNEL_H__ */
