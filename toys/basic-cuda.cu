#include <stdio.h>
// #include <cuda.h> 


__global__ 
void foo (float *farr) { 
  farr[0] = farr[1]; 
}


int main (void) {
  float *d_farr; 
  cudaMalloc(&d_farr, sizeof(float)*2); 
  
  foo<<<1, 1>>>(d_farr); 

  return 0;
}
