#include "FloydWarshallOMP.hpp"
#include <iostream>
#include <limits>

namespace FloydWarshallOMP {

void FloydWarshallOMP(float* matrix, int n){
    const auto INF = std::numeric_limits<float>::infinity();

    int i,j,k;

    for (k = 0; k < n; k++) {
        #pragma omp parallel for
        {
            for (i = 0; i < n; i++) {
                #pragma omp parallel for
                for (j = 0; j < n; j++) {
                    if (matrix[i * n +k] != INF &&
                        matrix[k * n + j] != INF &&
                        matrix[i * n + k] + matrix[k * n + j] < matrix[i * n + j]) {

                        matrix[i * n + j] = matrix[i * n + k] + matrix[k * n + j];
                    }
                }
            }
        }
    }
  }
}
