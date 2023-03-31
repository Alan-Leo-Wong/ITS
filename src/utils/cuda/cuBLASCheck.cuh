#pragma once

#include <stdio.h>
#include <iostream>
// v2一定要放在api前面
#include <cublas_v2.h>
#include <cublas_api.h>

#define CUBLAS_CHECK(call)                                                      \
    do                                                                        \
    {                                                                         \
        const cublasStatus_t error_code = call;                                  \
        if (error_code != CUBLAS_STATUS_SUCCESS)                                        \
        {                                                                     \
            fprintf(stderr, "cuBLAS Error:\n");                                          \
            fprintf(stderr, "    --File:       %s\n", __FILE__);                       \
            fprintf(stderr, "    --Line:       %d\n", __LINE__);                       \
            fprintf(stderr, "    --Error code: %d\n", error_code);                     \
            exit(EXIT_FAILURE);                                                          \
        }                                                                     \
    } while (0);