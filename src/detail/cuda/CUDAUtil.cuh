#pragma once

#include "CUDAMath.hpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <device_launch_parameters.h>

NAMESPACE_BEGIN(ITS)

#ifndef cuMIN
#  define cuMIN(x, y) ((x < y) ? x : y)
#endif // !MIN

#ifndef cuMAX
#  define cuMAX(x, y) ((x > y) ? x : y)
#endif // !MAX

// ȷ��������warpSize�����������Ա���ȷ�ؽ��������
#define PADDING_TO_WARP(nvRows) ((nvRows % 32 == 0) ? (nvRows) : (nvRows + 32 - (nvRows % 32)))

    template<class T>
    static inline __host__ void getOccupancyMaxPotentialBlockSize(const size_t &dataSize,
                                                                  int &minGridSize,
                                                                  int &blockSize,
                                                                  int &gridSize,
                                                                  T func,
                                                                  size_t dynamicSMemSize = 0,
                                                                  int blockSizeLimit = 0) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func, dynamicSMemSize, blockSizeLimit);
        gridSize = (dataSize + blockSize - 1) / blockSize;
    }

    template<class T>
    static inline __host__ void getOccupancyAvailableDynamicSMemPerBlock(const int &numBlocks,
                                                                         const int &blockSize,
                                                                         size_t &dynamicSmemSize,
                                                                         T func) {
        cudaOccupancyAvailableDynamicSMemPerBlock(&dynamicSmemSize, func, numBlocks, blockSize);
    }

    template<typename T>
    __host__ void resizeThrust(thrust::host_vector<T> &h_vec, const size_t &dataSize) {
        h_vec.resize(dataSize);
        h_vec.shrink_to_fit();
    }

    template<typename T>
    __host__ void resizeThrust(thrust::host_vector<T> &h_vec, const size_t &dataSize, const T &init) {
        h_vec.clear();
        h_vec.resize(dataSize, init);
        h_vec.shrink_to_fit();
    }

    template<typename T>
    __host__ void cleanupThrust(thrust::host_vector<T> &h_vec) {
        h_vec.clear();
        h_vec.shrink_to_fit();
    }

    template<typename T>
    __host__ void resizeThrust(thrust::device_vector<T> &d_vec, const size_t &dataSize) {
        d_vec.resize(dataSize);
        d_vec.shrink_to_fit();
    }

    template<typename T>
    __host__ void resizeThrust(thrust::device_vector<T> &d_vec, const size_t &dataSize, const T &init) {
        d_vec.clear();
        d_vec.resize(dataSize, init);
        d_vec.shrink_to_fit();
    }

    template<typename T>
    __host__ void cleanupThrust(thrust::device_vector<T> &d_vec) {
        d_vec.clear();
        d_vec.shrink_to_fit();
    }

    inline __host__ void getBlocksAndThreadsNum(const cudaDeviceProp &prop,
                                                const unsigned int &n, const int &maxBlocks,
                                                const int &maxThreads, int &blocks,
                                                int &threads) {
        // ���ʹ��n/2���߳���
        threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
        threads = isPow2(threads) ? nextPow2(threads) : threads;

        // ����maxBlocks�ɰ���ʵ�����񼶱�Ŀ��
        // (n + (threads * 2) - 1) / (threads * 2)��Ϊ��ʵ��һ���߳̿鼶��Ŀ��
        blocks = cuMIN(maxBlocks, (n + (threads * 2) - 1) / (threads * 2));

        if ((float) threads * blocks >
            (float) prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
            printf("n is too large, please choose a smaller number!\n");
            exit(EXIT_FAILURE);
        }

        if (blocks > prop.maxGridSize[0]) {
            printf("Grid size <%d> exceeds the device capability <%d>\n", blocks,
                   prop.maxGridSize[0]);
            const int t_blocks = blocks;
            const int t_threads = threads;
            while (blocks > prop.maxGridSize[0]) {
                if (threads * 2 <= maxThreads) {
                    threads *= 2;
                    blocks = (n + (threads * 2) - 1) / (threads * 2);
                } else {
                    break;
                }
            }
            printf("Set grid size as <%d> (original is <%d>), set block size as <%d> "
                   "(original is "
                   "<%d>)\n",
                   blocks, t_blocks, threads, t_threads);
        }
        //printf("-- Final grid size = %d, block size = %d\n", blocks, threads);
    }

    template<typename T>
    struct SharedMemory {
        __device__ inline operator T *() {
            extern __shared__ int __smem[];
            return (T *) __smem;
        }

        __device__ inline operator const T *() const {
            extern __shared__ int __smem[];
            return (T *) __smem;
        }
    };

NAMESPACE_END(ITS)
