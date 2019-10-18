#include "kernelBasic.hpp"
#include "kernelBlockedGlobal.hpp"
#include "kernelBlockedShared.hpp"
#include "FloydWarshall.hpp"
#include "Graph/GraphWeight.hpp"
#include <tuple>
#include <limits>
#include "time.h"
#include <iostream>
#include <fstream>

using matrix_t = float;

void saveMatrix(float** matrix, int n){

	std::ofstream pippo;
 	pippo.open("test.txt",  std::ios::out);
 	char s[]="test file";
	 for (int i = 0; i< n;i++){
        for (int j = 0; j< n;j++)
            pippo<< matrix[i][j]<< ";";
        pippo<<"\n";
    }
}

bool cmpFloats(float a, float b) {
    const float epsilon = 0.0000001f;
    float abs_a = std::abs(a);
    float abs_b = std::abs(b);
    float abs_diff = std::abs(a - b);

    if(a == b) {
        return true;
    } else if(a == 0 || b == 0 || abs_diff < std::numeric_limits<float>::min()) {

        return abs_diff < (epsilon * std::numeric_limits<float>::min());
    } else {
        return abs_diff / std::min(abs_a + abs_b, std::numeric_limits<float>::max()) < epsilon;
    }
}
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

	graph::GraphWeight<int, int, matrix_t> graph(graph::structure_prop::COO);
	graph.read(argv[1]);

	std::string graph_name = argv[1];

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

	auto matrix_input = new matrix_t[graph.nV()*graph.nV()];
    for (int i = 0; i < graph.nV() * graph.nV(); i++){
        matrix_input[i] = std::numeric_limits<matrix_t>::infinity();
    }
    for (int i = 0; i < graph.nE(); i++){
        auto index = graph.coo_ptr()[i];
        matrix_input[std::get<0>(index) * graph.nV() + std::get<1>(index)] = std::get<2>(index);
    }

	auto matrix_device = new matrix_t[graph.nV()*graph.nV()];

	int choice = 0;
	std::cout << "Choose one of the following kernels: " << std::endl;
	std::cout << "1: Basic Kernel " << std::endl;
	std::cout << "2: Blocked Kernel global memory version" << std::endl;
	std::cout << "3: Blocked Kernel shared memory version" << std::endl;
	std::cout << "4: All version" << std::endl;
	std::cout << "Choice: ";
	std::cin >> choice;

	while(choice < 1 || choice > 4){
		std::cout << "Bad choice! Please try again later.\n";
		std::cin >> choice;
	}

	int n_volte;
	std::cout << "How many time would you execute the kernel? " << std::endl;
	std::cin >> n_volte;
	while(n_volte < 1){
		std::cout << "The choice must be an integer >= 1.\n";
		std::cin >> n_volte;
	}


/*
------------------------------- 	SEQUENZIALE 	--------------------------------
*/
	std::cout << "\nSEQUENTIAL" << std::endl;

	auto start = std::chrono::system_clock::now();
		floyd_warshall::floyd_warshall(matrix, graph.nV());
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> host_diff = end - start;
  	double t_spent = std::chrono::duration_cast<std::chrono::microseconds>(host_diff).count()/1000000.0;
	std::cout << "\n";
  	std::cout << "TIMING SEQUENTIAL CODE: " << t_spent << "s" << std::endl;
	std::cout << "\n";

	switch (choice) {
		case 1: {
			std::cout << "BASIC KERNEL CONFIGURATION" << std::endl;

			double mediaSpeedUp = 0;
			double speedUp = 0;
			double maxSpeedUp = 0;
			double mintime = t_spent;
			int k;
			for(k = 0; k < n_volte; k++){

				std::cout << "-------------- " << (k+1) << " TIME -------------- " << std::endl;

				for (int i = 0; i < graph.nV() * graph.nV(); i++)
					matrix_device[i] = matrix_input[i];

				/*
				------------------------------- 	BASIC KERNEL CONFIGURATION 	--------------------------------
				*/
				double temp_gpu = kernelBasic::kernelBasic(matrix_device, graph.nV());
				speedUp = t_spent / temp_gpu;
				std::cout << "TEMPO GLOBAL KERNEL = " << temp_gpu  << std::endl;
				std::cout << "SPEED UP BASIC KERNEL = " << speedUp << "x" << std::endl;
				mediaSpeedUp += speedUp;
				if(temp_gpu < mintime){
					maxSpeedUp = speedUp;
					mintime = temp_gpu;
				}
				checkMatrix(matrix, matrix_device, graph.nV());
				std::cout << "\n" << std::endl;

			}

			std::cout << "TEMPO SEQUENZIALE BASIC KERNEL = " << t_spent << std::endl;
			std::cout << "TEMPO MINIMO PARALLELO BASIC KERNEL = " << mintime << std::endl;
			std::cout << "MEDIA BASIC KERNEL = " << mediaSpeedUp/n_volte << std::endl;
			std::cout << "MAX SPEEDUP BASIC KERNEL = " << maxSpeedUp << std::endl;
			break;
		}
	case 2:	{
		std::cout << "BLOCKED KERNEL GLOBAL MEMORY VERSION" << std::endl;

		double mediaSpeedUp = 0;
		double speedUp = 0;
		double maxSpeedUp = 0;
		double mintime = t_spent;
		int k;

		for(k = 0; k < n_volte; k++){

			std::cout << "-------------- " << (k+1) << " TIME -------------- " << std::endl;

			for (int i = 0; i < graph.nV() * graph.nV(); i++)
				matrix_device[i] = matrix_input[i];

			/*
			------------------------------- 	BLOCKED KERNEL CONFIGURATION GLOBAL MEMORY	--------------------------------
			*/
			double temp_gpu_block_global = kernelBlockedGlobal::kernelBlockedGlobal(matrix_device, graph.nV());
			speedUp = t_spent / temp_gpu_block_global;
			std::cout << "TEMPO GLOBAL KERNEL = " << temp_gpu_block_global  << std::endl;
			std::cout << "SPEED UP GLOBAL KERNEL = " << speedUp << " x" << std::endl;
			mediaSpeedUp += speedUp;
			if(temp_gpu_block_global < mintime){
				maxSpeedUp = speedUp;
				mintime = temp_gpu_block_global;
			}
			checkMatrix(matrix, matrix_device, graph.nV());

			std::cout << "\n" << std::endl;
		}

		std::cout << "TEMPO SEQUENZIALE GLOBAL KERNEL = " << t_spent << std::endl;
		std::cout << "TEMPO MINIMO PARALLELO GLOBAL KERNEL = " << mintime << std::endl;
		std::cout << "MEDIA SPEED UP GLOBAL KERNEL = " << mediaSpeedUp/n_volte << std::endl;
		std::cout << "MAX GLOBAL KERNEL = " << maxSpeedUp << "\n\n" << std::endl;
		break;
		}

	case 3:	{
		std::cout << "BLOCKED KERNEL SHARED MEMORY VERSION" << std::endl;

		double mediaSpeedUp = 0;
		double speedUp = 0;
		double maxSpeedUp = 0;
		double mintime = t_spent;
		int k;
		for(k = 0; k < n_volte; k++){

			std::cout << "-------------- " << (k+1) << " TIME -------------- " << std::endl;

			/*
			------------------------------- 	BLOCKED KERNEL CONFIGURATION 2D SHARED MEMORY	--------------------------------
			*/
			for (int i = 0; i < graph.nV() * graph.nV(); i++)
				matrix_device[i] = matrix_input[i];
			double temp_gpu_block_shared = kernelBlockedShared::kernelBlockedShared(matrix_device, graph.nV());
			speedUp = t_spent / temp_gpu_block_shared;
			std::cout << "TIMING KERNEL SHARED CODE: " << temp_gpu_block_shared << std::endl;
			std::cout << "SPEED UP KERNEL SHARED = " << speedUp << "x" << std::endl;
			mediaSpeedUp += speedUp;
			if(temp_gpu_block_shared < mintime){
				maxSpeedUp = speedUp;
				mintime = temp_gpu_block_shared;
			}
			checkMatrix(matrix, matrix_device, graph.nV());

			std::cout << "\n" << std::endl;
		}

		std::cout << "TEMPO SEQUENZIALE SHARED KERNEL = " << t_spent << std::endl;
		std::cout << "TEMPO MINIMO PARALLELO SHARED KERNEL = " << mintime << std::endl;
		std::cout << "MEDIA SHARED KERNEL = " << mediaSpeedUp/n_volte << std::endl;
		std::cout << "MAX SHARED KERNEL = " << maxSpeedUp << std::endl;

		break;
	}

	case 4:{

		std::cout << "BASIC KERNEL CONFIGURATION" << std::endl;

		double mediaSpeedUp = 0;
		double speedUp = 0;
		double maxSpeedUp = 0;
		double mintime = t_spent;
		int k;
		for(k = 0; k < n_volte; k++){

			std::cout << "-------------- " << (k+1) << " TIME -------------- " << std::endl;

			for (int i = 0; i < graph.nV() * graph.nV(); i++)
				matrix_device[i] = matrix_input[i];

			/*
			------------------------------- 	BASIC KERNEL CONFIGURATION 	--------------------------------
			*/
			double temp_gpu = kernelBasic::kernelBasic(matrix_device, graph.nV());
			speedUp = t_spent / temp_gpu;
			std::cout << "TEMPO GLOBAL KERNEL = " << temp_gpu  << std::endl;
			std::cout << "SPEED UP BASIC KERNEL = " << speedUp << "x" << std::endl;
			mediaSpeedUp += speedUp;
			if(temp_gpu < mintime){
				maxSpeedUp = speedUp;
				mintime = temp_gpu;
			}
			checkMatrix(matrix, matrix_device, graph.nV());
			std::cout << "\n" << std::endl;

		}

		std::cout << "TEMPO SEQUENZIALE BASIC KERNEL = " << t_spent << std::endl;
		std::cout << "TEMPO MINIMO PARALLELO BASIC KERNEL = " << mintime << std::endl;
		std::cout << "MEDIA BASIC KERNEL = " << mediaSpeedUp/n_volte << std::endl;
		std::cout << "MAX SPEEDUP BASIC KERNEL = " << maxSpeedUp << std::endl;

		std::cout << "\n\nBLOCKED KERNEL GLOBAL MEMORY VERSION" << std::endl;

		mediaSpeedUp = 0;
		speedUp = 0;
		maxSpeedUp = 0;
		mintime = t_spent;

		for(k = 0; k < n_volte; k++){

			std::cout << "-------------- " << (k+1) << " TIME -------------- " << std::endl;

			//printf("------------------------INIZIO COPIA----------------------------------------------------\n");
			for (int i = 0; i < graph.nV() * graph.nV(); i++)
				matrix_device[i] = matrix_input[i];

			/*
			------------------------------- 	BLOCKED KERNEL CONFIGURATION GLOBAL MEMORY	--------------------------------
			*/
			double temp_gpu_block_global = kernelBlockedGlobal::kernelBlockedGlobal(matrix_device, graph.nV());
			speedUp = t_spent / temp_gpu_block_global;
			std::cout << "TEMPO GLOBAL KERNEL = " << temp_gpu_block_global  << std::endl;
			std::cout << "SPEED UP GLOBAL KERNEL = " << speedUp << " x" << std::endl;
			mediaSpeedUp += speedUp;
			if(temp_gpu_block_global < mintime){
				maxSpeedUp = speedUp;
				mintime = temp_gpu_block_global;
			}
			checkMatrix(matrix, matrix_device, graph.nV());

			std::cout << "\n" << std::endl;
		}

		std::cout << "TEMPO SEQUENZIALE GLOBAL KERNEL = " << t_spent << std::endl;
		std::cout << "TEMPO MINIMO PARALLELO GLOBAL KERNEL = " << mintime << std::endl;
		std::cout << "MEDIA SPEED UP GLOBAL KERNEL = " << mediaSpeedUp/n_volte << std::endl;
		std::cout << "MAX GLOBAL KERNEL = " << maxSpeedUp << "\n\n" << std::endl;

		std::cout << "\n\nBLOCKED KERNEL SHARED MEMORY VERSION" << std::endl;

		mediaSpeedUp = 0;
		speedUp = 0;
		maxSpeedUp = 0;
		mintime = t_spent;
		for(k = 0; k < n_volte; k++){

			std::cout << "-------------- " << (k+1) << " TIME -------------- " << std::endl;

			/*
			------------------------------- 	BLOCKED KERNEL CONFIGURATION 2D SHARED MEMORY	--------------------------------
			*/
			for (int i = 0; i < graph.nV() * graph.nV(); i++)
				matrix_device[i] = matrix_input[i];
			double temp_gpu_block_shared = kernelBlockedShared::kernelBlockedShared(matrix_device, graph.nV());
			speedUp = t_spent / temp_gpu_block_shared;
			std::cout << "TIMING KERNEL SHARED CODE: " << temp_gpu_block_shared << std::endl;
			std::cout << "SPEED UP KERNEL SHARED = " << speedUp << "x" << std::endl;
			mediaSpeedUp += speedUp;
			if(temp_gpu_block_shared < mintime){
				maxSpeedUp = speedUp;
				mintime = temp_gpu_block_shared;
			}
			checkMatrix(matrix, matrix_device, graph.nV());

			std::cout << "\n" << std::endl;
		}

		std::cout << "TEMPO SEQUENZIALE SHARED KERNEL = " << t_spent << std::endl;
		std::cout << "TEMPO MINIMO PARALLELO SHARED KERNEL = " << mintime << std::endl;
		std::cout << "MEDIA SHARED KERNEL = " << mediaSpeedUp/n_volte << std::endl;
		std::cout << "MAX SHARED KERNEL = " << maxSpeedUp << std::endl;

		break;

	}

	default:{
  		break;
	}
	}

	for (int i = 0; i < graph.nV(); i++)
        delete[] matrix[i];

	delete[] matrix;
	delete[] matrix_device;
}
