
int initRands(dim3 threadGrid, dim3 blockGrid, curandState *state); 
int advanceTimestep(dim3 threadGrid, dim3 blockGrid, curandState *rands, int* states, int* states2, int* wait, int* maxStates, int N_x, float alpha);
void recordData(dim3 threadGrid, int blockGrid, int* states, int* states2, int N_x, float* d_up, float* d_down, int* d_upcount, int* d_downcount, int t);
void countStates(int numThreads, int numBlocks, int* states, int* blockTotals);
