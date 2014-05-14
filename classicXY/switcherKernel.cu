
#include <curand_kernel.h>
#include <stdio.h>
#include "states.h"

__device__ int getIndex(int t_x, int t_y)
{

    // convert a position on the lattice to an index in the full array
    // the size of the lattice is threadx * blockx, and if there are 
    // multiple reps the blocky id needs to be added to the index count
    int y_block = blockIdx.y / gridDim.x;
    int N_x = gridDim.x * blockDim.x;

    return __mul24(y_block,__mul24(N_x,N_x)) + t_x + __mul24(N_x, t_y);


}
        

__global__ void d_initRands(curandState *state)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int id = getIndex(index_x, index_y);

    /* Each thread gets same seed, a different sequence 
     *        number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

__global__ void d_updateStates(int* states, int* newstates, int* wait, int* maxStates, int N_x, curandState* d_rands, float alpha)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;

 //   printf ("id %d, thread number %d, %d block %d, %d\n", 0, threadIdx.x,threadIdx.y, blockIdx.x, blockIdx.y);
    int id = getIndex(index_x, index_y);

    int edges=8;
    int neigh[8][2] = { { 1, 1 }, { 1, 0 }, { 1, -1 } , { 0, 1 }, { 0, -1 }, { -1, -1 } , { -1, 0 }, { -1, 1 } };

    float p_states[NUM_STATES];
    for (int s=0;s<NUM_STATES;s++)
        p_states[s]=0.0f;

    for (int e=0;e<edges;e++)
    {
            int x_n = (((index_x + neigh[e][0]) % N_x) + N_x) % N_x;
            int y_n = (((index_y + neigh[e][1]) % N_x) + N_x) % N_x;

            int n2_id = getIndex(x_n, y_n);
    //printf ("tx %d, ty %d, n2 %d, ns %d, ps %f, edge %d\n", threadIdx.x,threadIdx.y, n2_id, states[n2_id], p_states[states[n2_id]], e);
   //         n2_id = 0;//getIndex(x_n, y_n);
            p_states[states[n2_id]]+=(1.0f/(float)edges);


    }
    float rnd = curand_uniform(&d_rands[id]);
    float slice = 0.0f;
    float rnd_state;
    float av_state = 0.0f;
    for (int s=0;s<NUM_STATES;s++)
        av_state += ((float)s)*p_states[s];

    for (int s=0;s<NUM_STATES;s++)
    {
        slice += p_states[s];
        if (rnd<slice)
        {
            rnd_state=(float)s;
            break;
        }

    }

    int newState = round(alpha*av_state+(1.0f-alpha)*rnd_state);
    if (states[id] > maxStates[id])
        maxStates[id] = states[id];


    newstates[id] = states[id];

    if (wait[id] == 0)
    {
        if (states[id] != newState)
        {
            newstates[id] = newState;
            wait[id] = WAIT_TIME;
        }
    }
    else
    {
        wait[id]--;
    }

    // */
    //debug
    //   newstates[id] = id;
}

int initRands(dim3 threadGrid, dim3 blockGrid, curandState *state) 
{
    d_initRands<<< blockGrid, threadGrid >>>(state);
    if (cudaSuccess != cudaGetLastError()) 
 {printf ("Error in %s, line %d: %s \n", __FILE__ , __LINE__, cudaGetErrorString( cudaGetLastError() ) ); return -1; }
    return 0;
}
int advanceTimestep(dim3 threadGrid, dim3 blockGrid, curandState *rands, int* states, int* newstates, int* wait, int* maxStates, int N_x, float alpha)
{
    d_updateStates<<< blockGrid, threadGrid >>>(states, newstates, wait, maxStates, N_x, rands, alpha);
    if (cudaSuccess != cudaGetLastError()) 
    {
        printf ("Error in %s, line %d: %s \n", __FILE__ , __LINE__, cudaGetErrorString( cudaGetLastError() ) ); 
        return -1; 
    }

    return 0;
}
