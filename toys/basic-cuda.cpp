#include <stdio.h>
 


// __global__ 
void foo (float *farr) { 
  farr[0] = farr[1]; 
}


int main (void) {
  float *d_farr; 
  // cudaMalloc(&d_farr, sizeof(float)*2);
  {// __set_CUDAConfig(1, 1); 
          
 foo(d_farr);}
           

  return 0;
}
