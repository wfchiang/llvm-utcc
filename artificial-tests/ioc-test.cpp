#include <stdio.h> 

int main (void) {
  char c = 0; 
  int i = 0; 
  unsigned int ui = 0; 
  float fp = 0; 

  printf("-- integer overflow -- \n"); 
  i = 1000000; 
  c = i; 

  printf("-- negative int to unsigned int -- \n"); 
  i = -1; 
  ui = i; 

  /*
  printf("-- negative float to unsigned int -- \n"); 
  fp = -1.0; 
  ui = fp; 
  */

  return 0; 
}
