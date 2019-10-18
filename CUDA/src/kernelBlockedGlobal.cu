#include <tuple>
#include <limits>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <cuda.h>
#include <fstream>
#include "kernelBlockedGlobal.hpp"

#define BLOCK_SIZE 32

const auto INF = std::numeric_limits<float>::infinity();

__device__
void block_calc_global(float* matrix, const int baseBlock, int n, const int idy, const int idx){

  const int index = idy * n + idx;
  if(idx < n && idy < n){
    float old = matrix[index];

  for(int j = 0; j < BLOCK_SIZE; j++){
    int k_index = BLOCK_SIZE * baseBlock + j;
    int index1 = idy * n + k_index;
    int index2 = k_index * n + idx;

    if(k_index < n && matrix[index1] != INF && matrix[index2] != INF){
      float sum = matrix[index1] + matrix[index2];
      if(sum < old){
        matrix[index] = old = sum;
      }
    }
    __syncthreads();
  }
  matrix[index] = old;
  }
}

//diagonale
__global__ void global_Phase1(float* matrix, int baseBlock, int n){

  const int idx = BLOCK_SIZE * baseBlock + threadIdx.x;
  const int idy = BLOCK_SIZE * baseBlock + threadIdx.y;

  block_calc_global(matrix, baseBlock, n, idy, idx);
}

//riga e colonna
__global__ void global_Phase2(float* matrix, int baseBlock, int n){

  if(blockIdx.x == baseBlock) return;

  int idx = BLOCK_SIZE * baseBlock + threadIdx.x;
  int idy = BLOCK_SIZE * baseBlock + threadIdx.y;

  if(blockIdx.y == 0) {
    idx = BLOCK_SIZE * blockIdx.x + threadIdx.x; //per i blocchi orizzontali cambio indice idx
  } else {
    idy = BLOCK_SIZE * blockIdx.x + threadIdx.y; //per i blocchi verticali cambio indice idy
  }

  block_calc_global(matrix, baseBlock, n,  idy, idx);

}

//rimanenti
__global__ void global_Phase3(float* matrix, int baseBlock, int n){

  if(blockIdx.x == baseBlock || blockIdx.y == baseBlock) {
    return;
  }

  const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
  const int idy = BLOCK_SIZE * blockIdx.y + threadIdx.y;

  block_calc_global(matrix, baseBlock, n, idy, idx);

}


namespace kernelBlockedGlobal {

  float kernelBlockedGlobal(float* matrix, int n){

    float* matrix_device;

    cudaEvent_t start, stop;

     cudaMalloc(&matrix_device, n * n * sizeof(float));

       cudaMemcpy(matrix_device, matrix, n * n * sizeof(float), cudaMemcpyHostToDevice);

       int numBlock = ceil(n/float(BLOCK_SIZE));
       dim3 Phase1(1,1,1);
       dim3 Phase2(numBlock, 2, 1);
       dim3 Phase3(numBlock, numBlock, 1);
       dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

       cudaEventCreate(&start);
       cudaEventCreate(&stop);

       float millis =0;
       cudaEventRecord(start);

       for(int baseBlock = 0; baseBlock < numBlock; baseBlock++) {
         global_Phase1<<<Phase1, dimBlock>>>(matrix_device, baseBlock, n);
         global_Phase2<<<Phase2, dimBlock>>>(matrix_device, baseBlock, n);
         global_Phase3<<<Phase3, dimBlock>>>(matrix_device, baseBlock, n);
        }
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&millis, start, stop);
      float timingCompute = millis/1000;

      cudaMemcpy(matrix, matrix_device, n * n * sizeof(float), cudaMemcpyDeviceToHost);

      cudaEventDestroy(start);
     	cudaEventDestroy(stop);

      cudaFree(matrix_device);

    return timingCompute;
    }
  }
