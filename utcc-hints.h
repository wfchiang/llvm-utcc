#include <assert.h>

void claimNonNegativeInt (int *in) {}
void claimNonNegativeUint (unsigned int *in) {} 
void claimNonNegativeFP32 (float *in) {} 
void claimNonNegativeFP64 (double *in) {}

struct NonNegativeXYZCoordinates {
  unsigned int x; 
  unsigned int y; 
  unsigned int z; 
}; 

struct float2 {
  float x; 
  float y; 
}; 

struct float4 {
  float w; 
  float x; 
  float y; 
  float z; 
}; 

struct uchar4 { 
  unsigned char w; 
  unsigned char x;
  unsigned char y; 
  unsigned char z; 
}; 

struct uint3 {
  unsigned int x; 
  unsigned int y; 
  unsigned int z; 
}; 

struct uint4 {
  unsigned int w; 
  unsigned int x;
  unsigned int y; 
  unsigned int z;
}; 

typedef struct NonNegativeXYZCoordinates NonNegativeXYZCoordinates; 
typedef struct float2 float2; 
typedef struct float4 float4; 
typedef struct uchar4 uchar4; 
typedef struct uint3 uint3; 
typedef struct uint4 uint4; 

typedef struct uint3 dim3; 

// Since utcc is a static analysis, the actual values are not important 
const struct NonNegativeXYZCoordinates gridDim = {0, 0, 0}; 
const struct NonNegativeXYZCoordinates blockDim = {0, 0, 0};
const struct NonNegativeXYZCoordinates blockIdx = {0, 0, 0}; 
const struct NonNegativeXYZCoordinates threadIdx = {0, 0, 0};

// some fake cuda math functions... 
// since utcc is a static analysis... the following functions don't have the actual code. 
// but it doesn't matter... for now 
double ceil (double in) { return in; } 
double floor (double in) { return in; } 
double fabs (double in) { return in; } 
double sqrt (double in) { return in; } 
double exp (double in) { return in; } 
// double length (double in) { return in; }
double min (double a, double b) { return (a >= b ? b : a); } 
double max (double a, double b) { return (a >= b ? a : b); } 

// some helper functions for float4
inline 
float2 float2float2Add (float2 a2, float2 b2) {
  float2 ret; 
  ret.x = a2.x + b2.x;
  ret.y = a2.y + b2.y; 
  return ret; 
}

inline 
float2 float2float2Sub (float2 a2, float2 b2) {
  float2 ret; 
  ret.x = a2.x - b2.x;
  ret.y = a2.y - b2.y; 
  return ret; 
}

inline 
float2 float2scalarMul (float2 a2, float fp) {
  float2 ret; 
  ret.x = a2.x * fp;
  ret.y = a2.y * fp; 
  return ret; 
}


// some helper functions for float4
inline
float4 float4float4Add (float4 a4, float4 b4) {
  float4 ret4; 
  ret4.w = a4.w + b4.w; 
  ret4.x = a4.x + b4.x; 
  ret4.y = a4.y + b4.y; 
  ret4.z = a4.z + b4.z; 
  return ret4; 
}

inline
float4 float4uchar4Add (float4 a4, uchar4 b4) {
  float4 f4; 
  f4.w = b4.w;
  f4.x = b4.x;
  f4.y = b4.y;
  f4.z = b4.z;
  return float4float4Add(a4, f4); 
}

inline
float4 float4float4Sub (float4 a4, float4 b4) {
  float4 ret4; 
  ret4.w = a4.w - b4.w; 
  ret4.x = a4.x - b4.x; 
  ret4.y = a4.y - b4.y; 
  ret4.z = a4.z - b4.z; 
  return ret4; 
}

inline
float4 float4scalarMul (float4 f4, float s) {
  float4 ret4; 
  ret4.w = f4.w * s; 
  ret4.x = f4.x * s; 
  ret4.y = f4.y * s; 
  ret4.z = f4.z * s; 
  return ret4; 
}

inline
float4 float4scalarDiv (float4 f4, float s) {
  float4 ret4; 
  ret4.w = f4.w / s; 
  ret4.x = f4.x / s; 
  ret4.y = f4.y / s; 
  ret4.z = f4.z / s; 
  return ret4; 
}

void 
utcc_assert(int cond) {
  assert(cond); 
}

