set(PROJECT "FloydWarshall")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.5)

include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
if (COMPILER_SUPPORTS_CXX14)
    add_compile_options("-std=c++14")  
else()
    message(FATAL_ERROR "Compiler ${CMAKE_CXX_COMPILER} has no C++14 support.")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# ------------------------------------------------------------------------------

add_compile_options("-O3")
    add_compile_options("-fopenmp")


include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_executable(Floyd_Warshall_OMP ${PROJECT_SOURCE_DIR}/main.cpp ${SRCS})
