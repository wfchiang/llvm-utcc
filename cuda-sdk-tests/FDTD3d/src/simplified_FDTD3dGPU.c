/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// #include "../inc/FDTD3dGPU.h"

// #include <iostream>
// #include <algorithm>
// #include <helper_functions.h>
// #include <helper_cuda.h>

// #include "../inc/FDTD3dGPUKernel.cuh"
#include "utcc-hints.h" 

#define size_t unsigned int 
#define k_blockDimX    32
#define k_blockDimMaxY 16
#define k_blockSizeMin 128
#define k_blockSizeMax (k_blockDimX * k_blockDimMaxY)

/*
bool getTargetDeviceGlobalMemSize(memsize_t *result, const int argc, const char **argv)
{
    int               deviceCount  = 0;
    int               targetDevice = 0;
[5~    size_t            memsize      = 0;

    // Get the number of CUDA enabled GPU devices
    // printf(" cudaGetDeviceCount\n");
    // checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    // Select target device (device 0 by default)
    // targetDevice = findCudaDevice(argc, (const char **)argv);

    // Query target device for maximum memory allocation
    // printf(" cudaGetDeviceProperties\n");
    // struct cudaDeviceProp deviceProp;
    // checkCudaErrors(cudaGetDeviceProperties(&deviceProp, targetDevice));

    // memsize = deviceProp.totalGlobalMem;

    // Save the result
    // *result = (memsize_t)memsize;
    // return true;
}
*/

      // bool 
char fdtdGPU(const int dimx, const int dimy, const int dimz, char userBlockSizeAssigned)
{
  // claimNonNegativeInt(&dimx); 
  // claimNonNegativeInt(&dimy); 

    // dim3              dimBlock;
    NonNegativeXYZCoordinates dimBlock;
    // dim3              dimGrid;
    NonNegativeXYZCoordinates dimGrid;

    /*
    cudaEvent_t profileStart = 0;
    cudaEvent_t profileEnd   = 0;
    const int profileTimesteps = timesteps - 1;
    */

    int userBlockSize;

    int in_size; 
    userBlockSize = ((userBlockSizeAssigned > 0) ? 
		     (min(max((in_size / k_blockDimX * k_blockDimX), k_blockSizeMin), 
			  k_blockSizeMax)) : 
		     k_blockSizeMax); 
    // Check the device limit on the number of threads
    // struct cudaFuncAttributes funcAttrib;
    // checkCudaErrors(cudaFuncGetAttributes(&funcAttrib, FiniteDifferencesKernel));

    // userBlockSize = MIN(userBlockSize, funcAttrib.maxThreadsPerBlock);

    // Set the block size

    dimBlock.x = k_blockDimX;
    unsigned int dbx = k_blockDimX; 

    // Visual Studio 2005 does not like std::min
    //    dimBlock.y = std::min<size_t>(userBlockSize / k_blockDimX, (size_t)k_blockDimMaxY);

    dimBlock.y = (((userBlockSize / k_blockDimX) < (size_t)k_blockDimMaxY) ? (userBlockSize / k_blockDimX) : (size_t)k_blockDimMaxY);
    unsigned int dby = (((userBlockSize / k_blockDimX) < (size_t)k_blockDimMaxY) ? (userBlockSize / k_blockDimX) : (size_t)k_blockDimMaxY); 
    // dimGrid.x  = (unsigned int)ceil((float)dimx / dimBlock.x);
    dimGrid.x  = (unsigned int)ceil((float)dimx / dbx);
    // dimGrid.y  = (unsigned int)ceil((float)dimy / dimBlock.y);
    dimGrid.y  = (unsigned int)ceil((float)dimy / dby);
    // printf(" set block size to %dx%d\n", dimBlock.x, dimBlock.y);
    // printf(" set grid size to %dx%d\n", dimGrid.x, dimGrid.y);

    // Check the block size is valid
    /*
    if (dimBlock.x < RADIUS || dimBlock.y < RADIUS)
    {
        printf("invalid block size, x (%d) and y (%d) must be >= radius (%d).\n", dimBlock.x, dimBlock.y, RADIUS);
        exit(EXIT_FAILURE);
    }
    */

    // Copy the input to the device input buffer
    // checkCudaErrors(cudaMemcpy(bufferIn + padding, input, volumeSize * sizeof(float), cudaMemcpyHostToDevice));

    // Copy the input to the device output buffer (actually only need the halo)
    // checkCudaErrors(cudaMemcpy(bufferOut + padding, input, volumeSize * sizeof(float), cudaMemcpyHostToDevice));

    // Copy the coefficients to the device coefficient buffer
    // checkCudaErrors(cudaMemcpyToSymbol(stencil, (void *)coeff, (radius + 1) * sizeof(float)));

    /*
    int it; 
    for (it = 0 ; it < timesteps ; it++)
    {
      // printf("\tt = %d ", it);

      // Launch the kernel
      // printf("launch kernel\n");
      // FiniteDifferencesKernel<<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
      
      // Toggle the buffers
      // Visual Studio 2005 does not like std::swap
      //    std::swap<float *>(bufferSrc, bufferDst);
      float *tmp = bufferDst;
      bufferDst = bufferSrc;
      bufferSrc = tmp;
    }
    */
    // printf("\n");

    // Wait for the kernel to complete
    // checkCudaErrors(cudaDeviceSynchronize());

    // Read the result back, result is in bufferSrc (after final toggle)
    // checkCudaErrors(cudaMemcpy(output, bufferSrc, volumeSize * sizeof(float), cudaMemcpyDeviceToHost));


    return 1;
}


int main (void) {
  return 0; 
}
