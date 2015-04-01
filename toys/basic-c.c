#include <stdio.h>
#include <assert.h>

int global_int = 0; 
float global_float = 0.0; 

#include "../utcc-hints.h" 

int main (void) {
  float f = -1.0; 
  // float f = 1.0; 
  // int i = (f >= 0) ? 1 : -1; 


  // assert(f >= 0.0); 
  unsigned int u = f; 

  return 0;
}
