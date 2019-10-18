#include <tuple>
#include <limits>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <cuda.h>
#include <fstream>
#include "kernelBlockedShared.hpp"

#define BLOCK_SIZE 32

const auto INF = std::numeric_limits<float>::infinity();


__device__
void block_calc_shared(float dest[][BLOCK_SIZE], float path1[][BLOCK_SIZE], float path2[][BLOCK_SIZE],
                const int baseBlock, const int n,
                const int idy, const int idx) {

    if(idx < n && idy < n) {

        float computePath = dest[threadIdx.y][threadIdx.x];

        for(int k=0;k<BLOCK_SIZE;k++) {
            int k_index = BLOCK_SIZE * baseBlock + k;


            float p1 = path1[threadIdx.y][k];    //LETTURA SULLO STESSO BANCO SHARED
            float p2 = path2[k][threadIdx.x];


            if(k_index < n && p1 != INF &&  p2 != INF) {
                float sumPath = p1 + p2;
                if(sumPath < computePath) {

                    computePath = sumPath;
                }
            }
            dest[threadIdx.y][threadIdx.x] = computePath;
            __syncthreads();
        }
    }
}

__global__
void shared_Phase1(float* matrix, const int baseBlock, const int n) {
    const int idx = BLOCK_SIZE * baseBlock + threadIdx.x;
    const int idy = BLOCK_SIZE * baseBlock + threadIdx.y;

    const int indexMatrix = idy * n + idx;
    __shared__ float sharedMatrix[BLOCK_SIZE][BLOCK_SIZE];

        //copio da global a shared
        //se dentro indice copio altrimenti metto INF
        if(idx < n && idy < n){
            sharedMatrix[threadIdx.y][threadIdx.x] = matrix[indexMatrix];
        }else{
            sharedMatrix[threadIdx.y][threadIdx.x] = INF;
        }

        __syncthreads();
        block_calc_shared(sharedMatrix,sharedMatrix,sharedMatrix, baseBlock, n,  idy, idx);

        //copio da shared a global

    if(idx < n && idy < n) {
        matrix[indexMatrix] = sharedMatrix[threadIdx.y][threadIdx.x];
    }
}

__global__
void shared_Phase2(float* matrix, const int baseBlock, const int n) {
    if(blockIdx.x == baseBlock) return;

    int idx = BLOCK_SIZE * baseBlock + threadIdx.x;
    int idy = BLOCK_SIZE * baseBlock + threadIdx.y;

    __shared__ float sharedMatrix[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedMatrixAux[BLOCK_SIZE][BLOCK_SIZE];

    int indexMatrix = idy * n + idx;
    //copio da global a shared
    //se dentro indice copio altrimenti metto INF
    if(idx < n && idy < n){
        sharedMatrix[threadIdx.y][threadIdx.x] = matrix[indexMatrix];
    }else{
        sharedMatrix[threadIdx.y][threadIdx.x] = INF;
    }

    if(blockIdx.y == 0) {
        //per i blocchi orizzontali cambio idx
        idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    } else {
        //per i blocchi verticali cambio idy
        idy = BLOCK_SIZE * blockIdx.x + threadIdx.y;
    }

    indexMatrix = idy * n + idx;

    //copio da global a shared seconda matrice
    //se dentro indice copio altrimenti metto INF
    if(idx < n && idy < n)
        sharedMatrixAux[threadIdx.y][threadIdx.x] = matrix[indexMatrix];
    else
        sharedMatrixAux[threadIdx.y][threadIdx.x] = INF;

    __syncthreads();
    if(blockIdx.y == 0) {
        block_calc_shared(sharedMatrixAux, sharedMatrix, sharedMatrixAux, baseBlock,n,idy,idx);
    } else {
        block_calc_shared(sharedMatrixAux, sharedMatrixAux, sharedMatrix, baseBlock,n,idy,idx);
    }

    //copio da shared a global
    if(idx < n && idy < n) {
        matrix[indexMatrix] = sharedMatrixAux[threadIdx.y][threadIdx.x];
    }
}

__global__
void shared_Phase3(float* matrix, const int b, const int n) {
    if(blockIdx.x == b || blockIdx.y == b) {
        return;
    }

    __shared__ float sharedMatrixCol[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedMatrixRow[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedMatrix[BLOCK_SIZE][BLOCK_SIZE];
    const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int idy = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    const int indexMatrix = idy * n + idx;

    //lettura da global

    if(idx < n && idy < n){
        sharedMatrix[threadIdx.y][threadIdx.x] = matrix[indexMatrix];
    }else{
        sharedMatrix[threadIdx.y][threadIdx.x] = INF;
    }

    const int indexMatrixCol = idy * n + (BLOCK_SIZE * b + threadIdx.x);
    if(indexMatrixCol < n*n){
        sharedMatrixCol[threadIdx.y][threadIdx.x] = matrix[indexMatrixCol];

    }else{
        sharedMatrixCol[threadIdx.y][threadIdx.x] = INF;

    }

    const int indexMatrixRow = idx + n * (BLOCK_SIZE * b + threadIdx.y);
    if(indexMatrixRow<n*n){
        sharedMatrixRow[threadIdx.y][threadIdx.x] = matrix[indexMatrixRow];
    }else{
        sharedMatrixRow[threadIdx.y][threadIdx.x] = INF;
    }

    __syncthreads();
    block_calc_shared(sharedMatrix, sharedMatrixCol, sharedMatrixRow, b,n, idy, idx);

    if(idx < n && idy < n) {
        matrix[indexMatrix] = sharedMatrix[threadIdx.y][threadIdx.x];
    }
}

namespace kernelBlockedShared {

    float kernelBlockedShared(float* matrix, int n) {

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
               shared_Phase1<<<Phase1, dimBlock>>>(matrix_device, baseBlock, n);
               shared_Phase2<<<Phase2, dimBlock>>>(matrix_device, baseBlock, n);
               shared_Phase3<<<Phase3, dimBlock>>>(matrix_device, baseBlock, n);
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
