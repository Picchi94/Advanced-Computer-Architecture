
#define CHECK_CUDA_ERROR                                                       \
    {                                                                          \
        cudaDeviceSynchronize();                                               \
        ::detail::getLastCudaError(__FILE__, __LINE__, __func__);              \
    }

#define SAFE_CALL(function)                                                    \
    {                                                                          \
        ::detail::safe_call(function, __FILE__, __LINE__, __func__);           \
    }

//------------------------------------------------------------------------------

namespace detail {

void getLastCudaError(const char* file, int line, const char* func_name);

void safe_call(cudaError_t error,
               const char* file,
               int         line,
               const char* func_name);

void cudaErrorHandler(cudaError_t error,
                      const char* error_message,
                      const char* file,
                      int         line,
                      const char* func_name);

} // namespace detail

#include "CheckError.i.cuh"
