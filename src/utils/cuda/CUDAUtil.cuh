#pragma once
#include <thrust\host_vector.h>
#include <thrust\device_vector.h>
#include <device_launch_parameters.h>

template<class T>
static inline __host__ void getOccupancyMaxPotentialBlockSize(const size_t& dataSize,
	int& minGridSize,
	int& blockSize,
	int& gridSize,
	T      func,
	size_t dynamicSMemSize = 0,
	int    blockSizeLimit = 0)
{
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func, dynamicSMemSize, blockSizeLimit);
	gridSize = (dataSize + blockSize - 1) / blockSize;
}

template<class T>
static inline __host__ void getOccupancyAvailableDynamicSMemPerBlock(const int& numBlocks,
	const int& blockSize,
	size_t& dynamicSmemSize,
	T      func)
{
	cudaOccupancyAvailableDynamicSMemPerBlock(&dynamicSmemSize, func, numBlocks, blockSize);
}

template<typename T>
__host__ void resizeThrust(thrust::host_vector<T>& h_vec, const size_t& dataSize)
{
	h_vec.resize(dataSize); h_vec.shrink_to_fit();
}

template<typename T>
__host__ void resizeThrust(thrust::host_vector<T>& h_vec, const size_t& dataSize, const T& init)
{
	h_vec.clear(); h_vec.resize(dataSize, init); h_vec.shrink_to_fit();
}

template<typename T>
__host__ void cleanupThrust(thrust::host_vector<T>& h_vec)
{
	h_vec.clear(); h_vec.shrink_to_fit();
}

template<typename T>
__host__ void resizeThrust(thrust::device_vector<T>& d_vec, const size_t& dataSize)
{
	d_vec.resize(dataSize); d_vec.shrink_to_fit();
}

template<typename T>
__host__ void resizeThrust(thrust::device_vector<T>& d_vec, const size_t& dataSize, const T& init)
{
	d_vec.clear(); d_vec.resize(dataSize, init); d_vec.shrink_to_fit();
}

template<typename T>
__host__ void cleanupThrust(thrust::device_vector<T>& d_vec)
{
	d_vec.clear(); d_vec.shrink_to_fit();
}