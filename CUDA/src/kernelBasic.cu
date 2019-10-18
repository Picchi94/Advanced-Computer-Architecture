#include <tuple>
#include <limits>
#include "CheckError.cuh"
#include <cuda_profiler_api.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <cuda.h>
#include <fstream>
#include "kernelBasic.hpp"

#define BLOCK_SIZE 32

const auto INF = std::numeric_limits<float>::infinity();

__global__ void computation(float* matrix, int k, int n){

  int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

  const unsigned int kj = k*n + j;
  const unsigned int ij = i*n + j;
  const unsigned int ik = i*n + k;

  if(i < n && j < n &&
          matrix[ik] != INF &&
          matrix[kj] != INF &&
          matrix[ik] + matrix[kj] < matrix[ij]) {
            matrix[ij] = matrix[ik] + matrix[kj];
      }
  }

namespace kernelBasic {

float kernelBasic(float* matrix, int n){

	float* matrix_device;

  cudaEvent_t start, stop;

	cudaError err = cudaMalloc(&matrix_device, n * n * sizeof(float));

  if(cudaSuccess != err){
    printf("error\n");
  }

	float msTime = 0;

	//copio i dati dall'host al deivce
	err = cudaMemcpy(matrix_device, matrix, n * n * sizeof(float), cudaMemcpyHostToDevice);

  if(cudaSuccess != err){
    printf("error\n");
  }

  //definisco i blocchi
dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, 1);
dim3 num_blocks(ceil(n/((float)BLOCK_SIZE) + 1), ceil(n/((float)BLOCK_SIZE) + 1), 1);

cudaEventCreate(&start);
cudaEventCreate(&stop);

float millis =0;
cudaEventRecord(start);
	//faccio eseguire i calcoli
	for(int i = 0; i < n; i++){
	   computation<<<num_blocks, block_size >>> (matrix_device, i, n);
  }

  cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millis, start, stop);
	float timingCompute = millis/1000;

	std::cout << "TIMING COMPUTE: " << timingCompute <<  "s\n";


	cudaMemcpy(matrix, matrix_device, n * n * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(matrix_device);

	return timingCompute;
	}
}
