

# define CUDA_CALL(x) do { if ((x) != cudaSuccess ) {printf ("Error in %s, line %d: %s \n", __FILE__ , __LINE__, cudaGetErrorString( x ) ); return EXIT_FAILURE ;}} while (0)                       
