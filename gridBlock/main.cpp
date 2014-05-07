
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

    // number of reps
    int numBlocks = 1;

    // length of grid
    int Nx = 32;
    int Nx2 = 0.5*Nx;
    int N = Nx * Nx;
    int N_ALL = N * numBlocks;

    const unsigned int shape[] = {1,Nx,Nx};


    dim3 threadGrid(Nx, Nx);
    curandState *devRands;
    CUDA_CALL(cudaMalloc((void **)&devRands, N_ALL * sizeof(curandState)));
    initRands(threadGrid, numBlocks, devRands);



    float alpha = 0.8;


    int* d_states;
    CUDA_CALL(cudaMalloc((void**)&d_states, sizeof(int) * N_ALL));

    int* h_states = new int[N_ALL];
    for (int i=0;i<N_ALL;i++)
        h_states[i] = 0;
    h_states[Nx2*Nx + Nx2] = state_vals[NUM_STATES - 1];
    CUDA_CALL(cudaMemcpy(d_states, h_states, (N_ALL) * sizeof(int), cudaMemcpyHostToDevice));
    int* d_wait;
    CUDA_CALL(cudaMalloc((void**)&d_wait, sizeof(int) * N_ALL));

    int* h_wait = new int[N_ALL];
    for (int i=0;i<N_ALL;i++)
        h_wait[i] = 0;
    h_wait[Nx2*Nx + Nx2] = 20;
    CUDA_CALL(cudaMemcpy(d_wait, h_wait, (N_ALL) * sizeof(int), cudaMemcpyHostToDevice));

    cnpy::npy_save("XY.npy",h_states,shape,3,"w");

    int* d_blockTotals;
    CUDA_CALL(cudaMalloc((void**)&d_blockTotals, sizeof(int) * numBlocks));
    CUDA_CALL(cudaMemset (d_blockTotals, 0, sizeof(int) * (numBlocks)));

    int* h_blockTotals = new int[numBlocks];

    for (int t=0;t<100;t++)
    {
        advanceTimestep(threadGrid, numBlocks, devRands, d_states, d_wait, Nx, alpha);
        //countStates(N, numBlocks, d_states, d_blockTotals);

        CUDA_CALL(cudaMemcpy(h_states, d_states, (N_ALL) * sizeof(int), cudaMemcpyDeviceToHost));
 /*           for (int i=0;i<Nx;i++)
            {
                for (int j=0;j<Nx;j++)
                    cout<<h_states[(j*Nx) + i]<<" ";
                cout<<endl;
            }
   */     //for (int i=0;i<N_ALL;i++)
        //    cout<<h_wait[i]<<endl;
        cout<<"*********"<<endl;
        cnpy::npy_save("XY.npy",h_states,shape,3,"a");

    }
    /*
        CUDA_CALL(cudaMemcpy(h_states, d_states, (N_ALL) * sizeof(float), cudaMemcpyDeviceToHost));
        for (int b=0;b<numBlocks;b++)
            for (int i=0;i<Nx;i++)
            {
                for (int j=0;j<Nx;j++)
                    cout<<h_states[(b*N) + (j*Nx) + i]<<" ";
                cout<<endl;
            }
            */
}
