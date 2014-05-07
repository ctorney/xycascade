
#include <curand_kernel.h>
#include <stdio.h>
#include "states.h"

__device__ int getIndex(int t_x, int t_y)
{
    // calculate full index from a grid position 
    int indx = __mul24(blockIdx.x,blockDim.x) + t_x;
    return __mul24(t_y, __mul24(gridDim.x, blockDim.x)) + indx;

}
        

__global__ void d_initRands(curandState *state)
{
    int id = getIndex(threadIdx.x, threadIdx.y);

    /* Each thread gets same seed, a different sequence 
     *        number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

__global__ void d_updateStates(int* states, int* wait, int N_x, curandState* d_rands, float alpha)
{
    int id = getIndex(threadIdx.x, threadIdx.y);
    int edges=8;
    int neigh[8][2] = { { 1, 1 }, { 1, 0 }, { 1, -1 } , { 0, 1 }, { 0, -1 }, { -1, -1 } , { -1, 0 }, { -1, 1 } };

    float p_states[NUM_STATES];
    for (int s=0;s<NUM_STATES;s++)
        p_states[s]=0.0f;
   
    for (int e=0;e<edges;e++)
    {
            int x_n = (((threadIdx.x + neigh[e][0]) % N_x) + N_x) % N_x;
            int y_n = (((threadIdx.y + neigh[e][1]) % N_x) + N_x) % N_x;

            int n2_id = getIndex(x_n, y_n);
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

    __syncthreads();

    if (wait[id] == 0)
    {
        if (states[id] != newState)
        {
            states[id] = newState;
            wait[id] = 5;
        }
    }
    else
        wait[id]--;
}
__global__ void d_recordData(int* states, int* net, int N_x, float* d_up, float* d_down, int* d_upcount, int* d_downcount, int t)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;

    int grid_width = gridDim.x * blockDim.x;
    int group_id = threadIdx.y * N_x + threadIdx.x;

    int N = N_x*N_x;

    if (group_id==0)
    {
        {
            int totalUp = 0;
            for (int i=0;i<N;i++)
                if (states[blockIdx.y * N + i] > 0)
                    totalUp++;
            

            int nowDown = 0;
            float pcDown = 0.0f;
            float pcUp = 0.0f;
            for (int i=0;i<N;i++)
            {
                int pop_id = blockIdx.y*N + i;
                int up =0;
                for (int j=0;j<N;j++)
                    if (net[pop_id * N + j]>0)
                        if (states[blockIdx.y * N + j]>0)
                            up++;
                if (states[blockIdx.y * N + i]>0)
                {
                    if ((float)up/3.0f>0.5)
                        pcUp += 1.0f;
                }
                else
                {
                    if ((float)up/3.0f<0.5)
                        pcDown += 1.0f ;
                }



            }
            pcUp /= totalUp;
            pcDown /= (N-totalUp);



            d_upcount[totalUp]+=1;
            int c = d_upcount[totalUp];
            if (c<1000)
            {
                d_down[totalUp] = (pcDown)/(float)c + (c-1)*d_down[totalUp]/(float)c;
                d_up[totalUp] = (pcUp)/(float)c + (c-1)*d_up[totalUp]/(float)c;
            }



   //         res[blockIdx.y] = counter/float(t+1) + t*res[blockIdx.y]/float(t+1);


        }

    
        //res[t * gridDim.y + blockIdx.y] = counter;
      //  if (t==0)
  //          res[blockIdx.y] = counter;
     //   else
   //         res[blockIdx.y] = counter/float(t+1) + t*res[blockIdx.y]/float(t+1);
    }
}
__global__ void block_sum(const int *input, int *per_block_results, const size_t n)
{
    return;
    extern __shared__ int sdata[];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // load input into __shared__ memory
    int x = 0;
    if(i < n)
    {
        x = input[i];
    }
    sdata[threadIdx.x] = x;
    __syncthreads();

    // contiguous range pattern
    for(int offset = blockDim.x / 2;
            offset > 0;
            offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            // add a partial sum upstream to our own
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }

        // wait until all threads in the block hav
        // updated their partial sums
        __syncthreads();
    }

    // thread 0 writes the final result
    if(threadIdx.x == 0)
    {
        per_block_results[blockIdx.x] = sdata[0];
    }
}

void initRands(dim3 threadGrid, int numBlocks, curandState *state) 
{
    d_initRands<<< numBlocks, threadGrid >>>(state);
    if (cudaSuccess != cudaGetLastError()) printf( "cuda error!\n" );
}
void advanceTimestep(dim3 threadGrid, int numBlocks, curandState *rands, int* states, int* wait, int N_x, float alpha)
{
    d_updateStates<<< numBlocks, threadGrid >>>(states, wait, N_x, rands, alpha);
    if (cudaSuccess != cudaGetLastError()) printf( "cuda error!\n" );

}
void countStates(int numThreads, int numBlocks, int* states, int* blockTotals)
{
    block_sum<<< numBlocks, numThreads >>>(states, blockTotals, numThreads);
    if (cudaSuccess != cudaGetLastError()) printf( "cuda error!\n" );

}
void recordData(dim3 threadGrid, int numBlocks, int* states, int* net, int N_x, float* d_up, float* d_down, int* d_upcount, int* d_downcount, int t)
{
    d_recordData<<< numBlocks, threadGrid >>>(states, net, N_x, d_up, d_down, d_upcount, d_downcount, t);
    if (cudaSuccess != cudaGetLastError()) printf( "cuda error!\n" );
}
