//#pragma once
//
//#include <stdio.h>
//#include <cuda_runtime.h>
//
//#define CUDA_CHECK(call)                                                      \
//    do                                                                        \
//    {                                                                         \
//        const cudaError_t error_code = call;                                  \
//        if (error_code != cudaSuccess)                                        \
//        {                                                                     \
//            fprintf(stderr, "CUDA Error:\n");                                          \
//            fprintf(stderr, "    --File:       %s\n", __FILE__);                       \
//            fprintf(stderr, "    --Line:       %d\n", __LINE__);                       \
//            fprintf(stderr, "    --Error code: %d\n", error_code);                     \
//            fprintf(stderr, "    --Error text: %s\n", cudaGetErrorString(error_code)); \
//            exit(EXIT_FAILURE);                                                          \
//        }                                                                     \
//    } while (0);
//
//#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)
//
//inline void __getLastCudaError(const char* errorMessage, const char* file, const int line)
//{
//	const cudaError_t error_code = cudaGetLastError();
//
//	if (error_code != cudaSuccess)
//	{
//        fprintf(stderr, "%s(%d) : getLastCudaError() CUDA Error :"
//            " %s : (%d) %s.\n",
//            file, line, errorMessage, static_cast<int>(error_code), cudaGetErrorString(error_code));
//        exit(EXIT_FAILURE);
//	}
//}