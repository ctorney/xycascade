
/*
 * XY model on a dynamic small world network 
*/

#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include "main.h"
#include "cuda_call.h"
#include <curand_kernel.h>
#include "switcherKernel.h"
#include "cnpy.h"
#include "states.h"

using namespace std;

int main(int argc, char** argv) 
{
    // size scaling, from 32x32 up to ....
    int scale = 4;
    // number of reps
    int reps = 1;
    static int success = 0;
    int ts;

    int blockX = scale;
    int blockY = scale * reps;
    int numBlocks = blockX * blockY;

    // length of grid
    int numThreads = 32;
    int Nx = numThreads * scale;
    int Nx2 = 0.5*Nx;
    int N = Nx * Nx;
    int N_ALL = N * reps;

    const unsigned int shape[] = {1,Nx,Nx};
    bool output = false;

    dim3 threadGrid(numThreads, numThreads);
    dim3 blockGrid(blockX, blockY);

    // device variables
    curandState *devRands;
    int* d_maxStates;
    int* d_states;
    int* d_states2;
    int* d_wait;

    CUDA_CALL(cudaMalloc((void **)&devRands, N_ALL * sizeof(curandState)));
    CUDA_CALL(cudaMalloc((void**)&d_maxStates, sizeof(int) * N_ALL));
    CUDA_CALL(cudaMalloc((void**)&d_states, sizeof(int) * N_ALL));
    CUDA_CALL(cudaMalloc((void**)&d_states2, sizeof(int) * N_ALL));
    CUDA_CALL(cudaMalloc((void**)&d_wait, sizeof(int) * N_ALL));


    // host variables
    int* h_states = new int[N_ALL];
    int* h_maxStates = new int[N_ALL];
    int* h_wait = new int[N_ALL];

    // initialize random numbers
    ts = initRands(threadGrid, blockGrid, devRands);
    if (ts!=success)
        return ts;
    float alpha = 0.0;
    for (int a = 0; a <= 300; a++)
    {
        alpha = 1.0f - (float)a * 0.001;
        int avMax = 0;
        int numRuns = 100;

        
        for (int av = 0; av < numRuns; av++)
        {

            // initialize states with one individual activated
            for (int i=0;i<N_ALL;i++)
            {
                h_states[i] = 0;
                h_wait[i] = 0;
            }

            for (int r = 0;r < reps; r++)
            {
                h_states[(r*N) + (Nx2-1)*Nx + Nx2-1] = state_vals[NUM_STATES - 1];
                h_wait[(r*N) + (Nx2-1)*Nx + Nx2-1] = WAIT_TIME;
            }


            // initialize device variables
            CUDA_CALL(cudaMemset(d_maxStates, 0, sizeof(int) * (N_ALL)));
            CUDA_CALL(cudaMemcpy(d_states, h_states, (N_ALL) * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(d_states2, h_states, (N_ALL) * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(d_wait, h_wait, (N_ALL) * sizeof(int), cudaMemcpyHostToDevice));


            if (output) 
                cnpy::npy_save("XY.npy",h_states,shape,3,"w");



            for (int t=0;t<5000;t++)
            {
                ts = advanceTimestep(threadGrid, blockGrid, devRands, d_states, d_states2, d_wait, d_maxStates, Nx, alpha);
                if (ts!=success)
                    return ts;
                ts = advanceTimestep(threadGrid, blockGrid, devRands, d_states2, d_states, d_wait, d_maxStates, Nx, alpha);
                if (ts!=success)
                    return ts;
                if (output)
                {
                    CUDA_CALL(cudaMemcpy(h_states, d_states, (N_ALL) * sizeof(int), cudaMemcpyDeviceToHost));
                    cnpy::npy_save("XY.npy",h_states,shape,3,"a");
                }

            }

            CUDA_CALL(cudaMemcpy(h_maxStates, d_maxStates, (N_ALL) * sizeof(int), cudaMemcpyDeviceToHost));

            for (int i=0;i<N_ALL;i++)
                avMax += (h_maxStates[i]>0);


        }
        cout<<alpha<<" "<<avMax/(float)(numRuns * N_ALL)<< endl; 
    }
}
