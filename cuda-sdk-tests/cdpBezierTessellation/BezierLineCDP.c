/**
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
// #include <cuda_runtime_api.h>
// #include <helper_cuda.h>
#include <string.h>

#include "utcc-hints.h"

#define sqrtf sqrt 

// __forceinline__ __device__ 
/*
inline 
float2 operator+(float2 a, float2 b)
{
    float2 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

// __forceinline__ __device__ 
inline float2 operator-(float2 a, float2 b)
{
    float2 c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    return c;
}

// __forceinline__ __device__ 
inline float2 operator*(float a, float2 b)
{
    float2 c;
    c.x = a * b.x;
    c.y = a * b.y;
    return c;
}
*/
// __forceinline__ __device__ 
/*
inline float length(float2 a)
{
    return sqrtf(a.x*a.x + a.y*a.y);
}
*/
#define length(f2) sqrtf(f2.x*f2.x + f2.y*f2.y) 

#define MAX_TESSELLATION 32
struct BezierLine
{
    float2 CP[3];
    float2 *vertexPos;
    int nVertices;
};

typedef struct BezierLine BezierLine; 

// __global__ 
void computeBezierLinePositions(int lidx, BezierLine *bLines, int nTessPoints)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx < nTessPoints)
    {
        float u = (float)idx/(float)(nTessPoints-1);
        float omu = 1.0f - u;

        float B3u[3];

        B3u[0] = omu*omu;
        B3u[1] = 2.0f*u*omu;
        B3u[2] = u*u;

        float2 position = {0,0};

	int i; 
        for (i = 0; i < 3; i++)
        {
	  // position = position + B3u[i] * bLines[lidx].CP[i];
	  position = float2float2Add(position, float2scalarMul(bLines[lidx].CP[i], B3u[i])); 
        }

        bLines[lidx].vertexPos[idx] = position;
    }
}

// __global__ 
void computeBezierLinesCDP(BezierLine *bLines, int nLines)
{
    int lidx = threadIdx.x + blockDim.x*blockIdx.x;

    if (lidx < nLines)
    {
      // float curvature = length(bLines[lidx].CP[1] - 0.5f*(bLines[lidx].CP[0] + bLines[lidx].CP[2]))/length(bLines[lidx].CP[2] - bLines[lidx].CP[0]);
        float curvature = 
	  length(float2float2Sub(bLines[lidx].CP[1], 
				 float2scalarMul(float2float2Add(bLines[lidx].CP[0], bLines[lidx].CP[2]), 
						 0.5f))) 
	  / length(float2float2Sub(bLines[lidx].CP[2], bLines[lidx].CP[0])); 
        int nTessPoints = min(max((int)(curvature*16.0f),4),MAX_TESSELLATION);
	unsigned int ui_nTessPoints = (float) nTessPoints; 

	//         if (bLines[lidx].vertexPos == NULL)
        {
            bLines[lidx].nVertices = nTessPoints;
            // cudaMalloc((void **)&bLines[lidx].vertexPos, nTessPoints*sizeof(float2));
	    bLines[lidx].vertexPos = (float2*) malloc(nTessPoints*sizeof(float2)); 
        }

        // computeBezierLinePositions<<<ceil((float)bLines[lidx].nVertices/32.0f), 32>>>(lidx, bLines, bLines[lidx].nVertices);
	unsigned int gdim = ceil((float)bLines[lidx].nVertices/32.0f); 
        computeBezierLinePositions(lidx, bLines, bLines[lidx].nVertices);
    }
}

// __global__ 
void freeVertexMem(BezierLine *bLines, int nLines)
{
    int lidx = threadIdx.x + blockDim.x*blockIdx.x;

    if (lidx < nLines)
      // cudaFree(bLines[lidx].vertexPos);
      ; 
}

// bool 
char checkCapableSM35Device(int argc, char** argv)
{
    // Get device properties
    // cudaDeviceProp properties;
    int device_count = 0, device = -1;
    /* 
    if(checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        device = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        
        cudaDeviceProp properties;
        checkCudaErrors(cudaGetDeviceProperties(&properties, device));
        
        if (properties.major > 3 || (properties.major == 3 && properties.minor >= 5))
        {
            printf("Running on GPU  %d (%s)\n", device , properties.name);
        }
        else
        {
            printf("cdpBezierTessellation requires GPU devices with compute SM 3.5 or higher.");
            printf("Current GPU device has compute SM %d.%d. Exiting...\n",properties.major, properties.minor);
            return false;
        }

    }
    else
    {
    
        checkCudaErrors(cudaGetDeviceCount(&device_count));

        for (int i=0; i < device_count; ++i)
        {
            checkCudaErrors(cudaGetDeviceProperties(&properties, i));

            if (properties.major > 3 || (properties.major == 3 && properties.minor >= 5))
            {
                device = i;
                printf("Running on GPU %d (%s)\n", i, properties.name);
                break;
            }

            printf("GPU %d %s does not support CUDA Dynamic Parallelism\n", i, properties.name);
        }
    }
    if (device == -1)
    {
        fprintf(stderr, "cdpBezierTessellation requires GPU devices with compute SM 3.5 or higher.  Exiting...\n");
        return false;
    }
    */
    // return true;
    return 1; 
}


#define N_LINES 256
#define BLOCK_DIM 64
#define RAND_MAX 1000000
#define EXIT_SUCCESS 0

int main(int argc, char **argv)
{
  // BezierLine *bLines_h = new BezierLine[N_LINES];
  BezierLine bLines_h [N_LINES];

    float2 last = {0,0};

    int i, j; 
    for (i = 0; i < N_LINES; i++)
    {
        bLines_h[i].CP[0] = last;

        for (j = 1; j < 3; j++)
        {
            bLines_h[i].CP[j].x = (float)rand()/(float)RAND_MAX;
            bLines_h[i].CP[j].y = (float)rand()/(float)RAND_MAX;
        }

        last = bLines_h[i].CP[2];
        bLines_h[i].vertexPos = NULL;
        bLines_h[i].nVertices = 0;
    }

    /*
    if (checkCapableSM35Device(argc, argv) == false)
    {
        exit(EXIT_SUCCESS);
    }
    */

    BezierLine *bLines_d; 
    // checkCudaErrors(cudaMalloc((void **)&bLines_d, N_LINES*sizeof(BezierLine)));
    bLines_d = (BezierLine*) malloc(N_LINES*sizeof(BezierLine)); 
    // checkCudaErrors(cudaMemcpy(bLines_d, bLines_h, N_LINES*sizeof(BezierLine), cudaMemcpyHostToDevice));
    // printf("Computing Bezier Lines (CUDA Dynamic Parallelism Version) ... ");
    // computeBezierLinesCDP<<< (unsigned int)ceil((float)N_LINES/(float)BLOCK_DIM), BLOCK_DIM >>>(bLines_d, N_LINES);
    int gdim = (unsigned int)ceil((float)N_LINES/(float)BLOCK_DIM); 
    computeBezierLinesCDP(bLines_d, N_LINES);
    // printf("Done!\n");

    //Do something to draw the lines here

    // freeVertexMem<<< (unsigned int)ceil((float)N_LINES/(float)BLOCK_DIM), BLOCK_DIM >>>(bLines_d, N_LINES);
    // checkCudaErrors(cudaFree(bLines_d));
    // delete[] bLines_h;

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exitsits
    // cudaDeviceReset();

    exit(EXIT_SUCCESS);
}
