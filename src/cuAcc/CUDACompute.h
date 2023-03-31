#pragma once
/*
* 一些公用 CUDA 计算
*/
#include "..\utils\cuda\CUDACheck.cuh"
#include "..\utils\cuda\cuBLASCheck.cuh"

void launch_BLASRowSumReduce(const cudaStream_t& stream,
	const int& rows,
	const int& columns,
	double* d_matrix,
	double* d_res);

//void accIntersection();