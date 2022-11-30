#ifndef __BFS_H__
#define __BFS_H__

#include <cuda_runtime.h>

#include "app/base_task.h"
#include "common/include/STGM.h"

struct Node {
    int starting;
    int no_of_edges;
};

typedef struct {
    // host vars
    int* h_cost;
    // device vars
    Node* d_graph_nodes;
    int* d_graph_edges;
    bool* d_graph_mask;
    bool* d_updating_graph_mask;
    bool* d_graph_visited;
    int* d_cost;
    bool *d_over=0;
    STGM_DEFINE_VARS();
}bfs_plan;

/* 
    For each task instance, we assume the matrix size doesn't change
*/
class BFS_TASK : public BASE_TASK {
    public:
        BFS_TASK(char*);
        ~BFS_TASK();
        /* functions */
        // init, run once at the system power on
        void taskInit();
        void taskInitDevice(unsigned int);
        // finish, run once at the system power off
        void taskFinish();
        // TODO: probably need to extract memcpy to another function if other works on the cpu is too much
        void taskPreGPU(cudaStream_t*, unsigned int); 
        void taskPostGPU(unsigned int _id); 
        void taskRunOnGPU(unsigned int, std::vector<int> &);

        /* variables  */
        cudaStream_t* the_stream;

        int no_of_nodes = 0;
        int edge_list_size = 0;
        FILE *fp;
        int source = 0;

        dim3 DimGrid, DimBlock;
        bfs_plan plan[2];
        Node* h_graph_nodes;
        bool *h_graph_mask;
        bool *h_updating_graph_mask;
        bool *h_graph_visited;
        int* h_graph_edges;
};

#endif /* __BFS_H__ */
