#include <stdio.h> 
#include "utcc-hints.h" 

// __device__ 
float fAdd (float arg0, float arg1) {
  return arg0 + arg1; 
}

// __global__ 
void bar (float *data, unsigned int *results) {
  int myid = threadIdx.x; 
  float fp0 = data[myid]; 
  float fp1 = data[myid + blockDim.x]; 
  // float fp2 = fAdd(fp0, fp1); 
  float fp2 = fabs(fAdd(fp0, fp1)); 
  
  results[myid] = (unsigned int) ceil(fp2); 
}

int main (void) {
  return 0;
} 
