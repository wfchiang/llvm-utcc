#include <stdio.h> 
#include "utcc-hints.h" 

// __global__ 
void foo (float *data, unsigned int *results) {
  int myid = threadIdx.x; 
  float fp0 = data[myid * 4 + 0]; 
  float fp1 = data[myid * 4 + 1]; 
  float fp2 = data[myid * 4 + 2]; 
  float fp3 = data[myid * 4 + 3]; 

  claimNonNegativeFP32(&fp0); 
  claimNonNegativeFP32(&fp1); 
  claimNonNegativeFP32(&fp2); 
  claimNonNegativeFP32(&fp3); 
  
  results[myid] = (unsigned int) ( myid < 256 ? (fp0 + fp1) : (fp2 - fp3) ); 
}

int main (void) {
  return 0;
} 
