set(PROJECT "FloydWarshall")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.5)
find_package(CUDA QUIET REQUIRED)

include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
if (COMPILER_SUPPORTS_CXX14)
    add_compile_options("-std=c++14")
else()
    message(FATAL_ERROR "Compiler ${CMAKE_CXX_COMPILER} has no C++14 support.")
endif()


# ------------------------------------------------------------------------------

add_compile_options("-O3")

include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE CU_SRCS ${PROJECT_SOURCE_DIR}/src/*.cu)
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_52,code=sm_52")
#list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30")
#list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_61,code=sm_61")
list(APPEND CUDA_NVCC_FLAGS "-w")
list(APPEND CUDA_NVCC_FLAGS "-g")
list(APPEND CUDA_NVCC_FLAGS "-lineinfo")
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")

add_library(floydwarshall STATIC ${PROJECT_SOURCE_DIR}/main.cpp ${SRCS})
cuda_add_executable(Floyd_Warshall_Cuda ${CU_SRCS})
target_link_libraries(Floyd_Warshall_Cuda floydwarshall)
#add_executable(floydwarshall ${PROJECT_SOURCE_DIR}/main.cpp ${SRCS})
