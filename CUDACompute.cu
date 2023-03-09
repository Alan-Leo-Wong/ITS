//#include "CUDACompute.h"
//#include "utils.hpp"
//#include <cuda/std/limits>
//
//__global__ void intersectionKernel(const MXd d_V,
//	const V2i* d_modelEdges,
//	const OctreeNode** d_leafNodes,
//	const int edgesNum,
//	const int leafNodesNum,
//	V3d* d_intersections)
//{
//	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
//	const int ty = blockIdx.y * blockDim.y + threadIdx.y;
//
//	if (tx < edgesNum && ty < leafNodesNum)
//	{
//		Eigen::Vector2i e = d_modelEdges[tx];
//		V3d p1 = d_V.row(e.x());
//		V3d dir = d_V.row(e.y()) - p1;
//
//		auto node = d_leafNodes[ty];
//
//		V3d leftBottomBackCorner = node->boundary.first;
//
//		double DINF = cuda::std::numeric_limits<double>::max();
//
//		// bottom plane
//		double bottom_t = DINF;
//		if (dir.z() != 0)
//			bottom_t = (leftBottomBackCorner.z() - p1.z()) / dir.z();
//		// left plane
//		double left_t = DINF;
//		if (dir.y() != 0)
//			left_t = (leftBottomBackCorner.y() - p1.y()) / dir.y();
//		// back plane
//		double back_t = DINF;
//		if (dir.x() != 0)
//			back_t = (leftBottomBackCorner.x() - p1.x()) / dir.x();
//
//		int id = 0;
//		if (isInRange(.0, 1.0, bottom_t))
//		{
//			d_intersections[tx * 3 + id] = p1 + bottom_t * dir;
//			++id;
//		}
//		if (isInRange(.0, 1.0, left_t))
//		{
//			d_intersections[tx * 3 + id] = p1 + left_t * dir;
//			++id;
//		}
//		if (isInRange(.0, 1.0, back_t))
//		{
//			d_intersections[tx * 3 + id] = p1 + back_t * dir;
//			++id;
//		}
//	}
//}
//
//void accIntersection()
//{
//
//}