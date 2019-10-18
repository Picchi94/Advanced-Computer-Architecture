#include "FloydWarshall.hpp"
#include "FloydWarshallOMP.hpp"
#include "Graph/GraphWeight.hpp"
#include <tuple>
#include "time.h"
#include <limits>
#include <omp.h>

using matrix_t = float;

bool cmpFloats2(float a, float b) {
    const float abs_a = std::abs(a);
    const float abs_b = std::abs(b);
    const float epsilon = 0.000001f;
    float diff = std::abs(abs_a - abs_b);
    if(diff / std::max(abs_a, abs_b) > epsilon) {
        return false;
    }
    return true;
}

void checkMatrix(float** matrix, float* matrix_device, int n){
	bool correct = true;
	for (int i = 0; i < n; i++) {
				for(int j=0;j<n; j++) {
						if(cmpFloats2(matrix[i][j],matrix_device[i*n + j]) == false) {
							correct = false;
								std::cerr.precision(std::numeric_limits<float>::max_digits10);
								std::cerr << "wrong result at: ("
												<< (i) << ", " << (j) << ")"
												<< "\nj:   " << j
												<< "\nhost:   " << matrix[i][j]
												<< "\ndevice: " << matrix_device[i * n + j] << "\n\n";
								std::exit(EXIT_FAILURE);
						}
		}
	}
	if(correct){
		std::cout << "CORRECT	"<< std::endl;
	}
}

int main(int argc, char* argv[]) {

    if (argc != 2)
        return EXIT_FAILURE;

    int n_volte;
	std::cout << "How many time would you execute the OMP version? " << std::endl;
  std::cin >> n_volte;
	while(n_volte < 1){
		std::cout << "Bad choice! The choice must be an integer >= 1.\n";
		std::cin >> n_volte;
	}

    graph::GraphWeight<int, int, matrix_t> graph(graph::structure_prop::COO);
    graph.read(argv[1]);

    auto matrix = new matrix_t*[graph.nV()];
    for (int i = 0; i < graph.nV(); i++) {
        matrix[i] = new matrix_t[graph.nV()];
        std::fill(matrix[i], matrix[i] + graph.nV(),
                  std::numeric_limits<matrix_t>::infinity());
    }
    for (int i = 0; i < graph.nE(); i++) {
        auto index = graph.coo_ptr()[i];
        matrix[std::get<0>(index)][std::get<1>(index)] = std::get<2>(index);
    }

    //-------------------------SEQUENZIALE--------------------------------------

    auto start = std::chrono::system_clock::now();
      floyd_warshall::floyd_warshall(matrix, graph.nV());
    auto end = std::chrono::system_clock::now();


    std::chrono::duration<double> host_diff = end - start;
    double t_spent = std::chrono::duration_cast<std::chrono::microseconds>(host_diff).count()/1000000.0;
  	std::cout << "\n";
    std::cout << "TIMING SEQUENTIAL CODE: " << t_spent << "s" << std::endl;
  	std::cout << "\n";

    //-------------------------OMP PARELLEL-------------------------------------

    int max_thread = omp_get_max_threads();

    std::cout << "Max Threads: " << max_thread << std::endl;

    auto matrixOMP = new matrix_t[graph.nV()*graph.nV()];

    auto matrix_input = new matrix_t[graph.nV()*graph.nV()];
    for (int i = 0; i < graph.nV() * graph.nV(); i++){
        matrix_input[i] = std::numeric_limits<matrix_t>::infinity();
    }
    for (int i = 0; i < graph.nE(); i++){
        auto index = graph.coo_ptr()[i];
        matrix_input[std::get<0>(index) * graph.nV() + std::get<1>(index)] = std::get<2>(index);
    }

    double mediaSpeedUp = 0;
    double speedUp = 0;
    double maxSpeedUp = 0;
    double mintime = t_spent;
    int k;
    for(k = 0; k < n_volte; k++){

        std::cout << "-------------- " << (k+1) << " TIME -------------- " << std::endl;

        for (int i = 0; i < graph.nV() * graph.nV(); i++)
            matrixOMP[i] = matrix_input[i];

        auto startOMP = std::chrono::system_clock::now();
        FloydWarshallOMP::FloydWarshallOMP(matrixOMP, graph.nV());
        auto stopOMP = std::chrono::system_clock::now();

        std::chrono::duration<double> omp_diff = stopOMP - startOMP;
        double t_spent_omp = std::chrono::duration_cast<std::chrono::microseconds>(omp_diff).count()/1000000.0;
        speedUp = t_spent / t_spent_omp;
        std::cout << "TEMPO OMP = " << t_spent_omp  << std::endl;
        std::cout << "SPEED UP BASIC KERNEL = " << speedUp << "x" << std::endl;
        mediaSpeedUp += speedUp;
        if(t_spent_omp < mintime){
            maxSpeedUp = speedUp;
            mintime = t_spent_omp;
        }
        checkMatrix(matrix, matrixOMP, graph.nV());
        std::cout << "\n" << std::endl;

    }

    std::cout << "TEMPO SEQUENZIALE = " << t_spent << std::endl;
    std::cout << "TEMPO MINIMO PARALLELO OMP = " << mintime << std::endl;
    std::cout << "MEDIA BASIC OMP = " << mediaSpeedUp/n_volte << std::endl;
    std::cout << "MAX SPEEDUP OMP = " << maxSpeedUp << std::endl;

    //--------------------------------------------------------------------------

    for (int i = 0; i < graph.nV(); i++)
        delete[] matrix[i];

    delete[] matrix;
    delete[] matrixOMP;
}
