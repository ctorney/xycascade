
void initRands(dim3 threadGrid, int blockGrid, curandState *state); 
void advanceTimestep(dim3 threadGrid, int numBlocks, curandState *rands, int* states, int* wait, int N_x, float alpha);
void recordData(dim3 threadGrid, int blockGrid, int* states, int* states2, int N_x, float* d_up, float* d_down, int* d_upcount, int* d_downcount, int t);
void countStates(int numThreads, int numBlocks, int* states, int* blockTotals);
