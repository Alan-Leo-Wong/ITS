#pragma once
/*
* һЩ���� CUDA ����
*/
#include "..\utils\cuda\CUDACheck.cuh"
#include "..\utils\cuda\cuBLASCheck.cuh"

void launch_BLASRowSumReduce(const cudaStream_t& stream,
	const int& rows,
	const int& columns,
	double* d_matrix,
	double* d_res);

//void accIntersection();