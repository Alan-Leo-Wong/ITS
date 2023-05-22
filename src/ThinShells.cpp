#include "ThinShells.h"
#include "MortonLUT.h"
#include "BSpline.hpp"
#include "utils/IO.hpp"
#include "utils/Timer.hpp"
#include "utils/Common.hpp"
#include "utils/String.hpp"
#include "cuAcc/CUDACompute.h"
#include "utils/cuda/CUDAMath.hpp"
#include "cuAcc/MarchingCubes/MarchingCubes.h"
#include <omp.h>
#include <queue>
#include <iomanip>
#include <numeric>
#include <Eigen/Sparse>
#include <igl/signed_distance.h>

//////////////////////
//  Create  Shells  //
//////////////////////
inline int parallelAxis(const V3d& p1, const V3d& p2)
{
	if (fabs(p1.y() - p2.y()) < 1e-9 && fabs(p1.z() - p2.z()) < 1e-9) return 1; // 与x轴平行
	else if (fabs(p1.x() - p2.x()) < 1e-9 && fabs(p1.z() - p2.z()) < 1e-9) return 2; // 与y轴平行
	else return 3; // 与z轴平行
};

inline V3i ThinShells::getPointDis(const V3d& vert, const V3d& origin, const V3d& width) const
{
	return ((vert - origin).array() / width.array()).cast<int>();
}

inline V3i ThinShells::getPointDis(const V3d& vert, const V3d& origin, const double& width) const
{
	return ((vert - origin).array() / width).cast<int>();
}

inline void ThinShells::cpIntersectionPoints()
{
	vector<V2i> modelEdges = extractEdges();
	uint nModelEdges = modelEdges.size();

	const size_t& numFineNodes = svo.numFineNodes;
	cout << "-- Number of level-0 nodes: " << numFineNodes << endl;;

	const vector<SVONode>& nodeArray = svo.svoNodeArray;
	const vector<node_edge_type>& fineNodeEdges = svo.fineNodeEdgeArray;

	// 只需要求三角形与最底层节点的交点

	// 三角形的边与node面交， 因为隐式B样条基定义在了left/bottom/back corner上， 所以与节点只需要求与这三个面的交即可
	std::cout << "1. Computing the intersections between mesh EDGES and nodes...\n";

	TimerInterface* timer = nullptr;
	createTimer(&timer);
	startTimer(&timer);

	vector<size_t> fineNodeIdx(svo.numFineNodes);
	std::iota(fineNodeIdx.begin(), fineNodeIdx.end(), 0);
	std::transform(nodeArray.begin(), nodeArray.begin() + svo.numFineNodes,
		fineNodeIdx.begin(), std::inserter(morton2FineNodeIdx, morton2FineNodeIdx.end()),
		[](const SVONode& node, const size_t& idx) {
			return std::make_pair(node.mortonCode, idx);
		});

#pragma omp parallel
	for (int i = 0; i < nModelEdges; i++)
	{
		std::vector<V3d> edge_vec_private;
		//std::vector<std::pair<V3d, uint32_t>> edge_morton_vec_private;

		Eigen::Vector2i e = modelEdges[i];
		V3d p1 = m_V.row(e.x()); V3d p2 = m_V.row(e.y());
		//if (!isLess(p1, p2, std::less<V3d>())) std::swap(p1, p2);

		V3d modelEdgeDir = p2 - p1;

		V3i dis1 = getPointDis(p1, modelOrigin, V3d(voxelWidth, voxelWidth, voxelWidth));
		V3i dis2 = getPointDis(p2, modelOrigin, V3d(voxelWidth, voxelWidth, voxelWidth));

		V3i min_dis = clamp(vmini(dis1, dis2).array() - 1, V3i(0, 0, 0), svo_gridSize.array() - 1);
		V3i max_dis = clamp(vmaxi(dis1, dis2).array() + 1, V3i(0, 0, 0), svo_gridSize.array() - 1);

		//#pragma omp for nowait collapse(3)
#pragma omp for nowait
		for (int z = min_dis.z(); z <= max_dis.z(); ++z)
		{
			for (int y = min_dis.y(); y <= max_dis.y(); ++y)
			{
				for (int x = min_dis.x(); x <= max_dis.x(); ++x)
				{
					uint32_t nodeMorton = morton::mortonEncode_LUT((uint16_t)x, (uint16_t)y, (uint16_t)z);
					if (morton2FineNodeIdx.find(nodeMorton) == morton2FineNodeIdx.end()) continue;
					V3d lbbCorner = nodeArray[morton2FineNodeIdx.at(nodeMorton)].origin; // at() is thread safe
					double width = nodeArray[morton2FineNodeIdx.at(nodeMorton)].width;

					// back plane
					double back_t = DINF;
					if (modelEdgeDir.x() != 0) back_t = (lbbCorner.x() - p1.x()) / modelEdgeDir.x();
					// left plane
					double left_t = DINF;
					if (modelEdgeDir.y() != 0) left_t = (lbbCorner.y() - p1.y()) / modelEdgeDir.y();
					// bottom plane
					double bottom_t = DINF;
					if (modelEdgeDir.z() != 0) bottom_t = (lbbCorner.z() - p1.z()) / modelEdgeDir.z();

					if (isInRange(.0, 1.0, back_t) &&
						isInRange(lbbCorner.y(), lbbCorner.y() + width, (p1 + back_t * modelEdgeDir).y()) &&
						isInRange(lbbCorner.z(), lbbCorner.z() + width, (p1 + back_t * modelEdgeDir).z()))
					{
						//edgeInterPoints.emplace_back(p1 + back_t * modelEdgeDir);
						edge_vec_private.emplace_back(p1 + back_t * modelEdgeDir);
						//edge_morton_vec_private.emplace_back(std::make_pair(p1 + back_t * modelEdgeDir, nodeMorton));
					}
					if (isInRange(.0, 1.0, left_t) &&
						isInRange(lbbCorner.x(), lbbCorner.x() + width, (p1 + left_t * modelEdgeDir).x()) &&
						isInRange(lbbCorner.z(), lbbCorner.z() + width, (p1 + left_t * modelEdgeDir).z()))
					{
						//edgeInterPoints.emplace_back(p1 + left_t * modelEdgeDir);
						edge_vec_private.emplace_back(p1 + left_t * modelEdgeDir);
						//edge_morton_vec_private.emplace_back(std::make_pair(p1 + back_t * modelEdgeDir, nodeMorton));
					}
					if (isInRange(.0, 1.0, bottom_t) &&
						isInRange(lbbCorner.x(), lbbCorner.x() + width, (p1 + bottom_t * modelEdgeDir).x()) &&
						isInRange(lbbCorner.y(), lbbCorner.y() + width, (p1 + bottom_t * modelEdgeDir).y()))
					{
						//edgeInterPoints.emplace_back(p1 + bottom_t * modelEdgeDir);
						edge_vec_private.emplace_back(p1 + bottom_t * modelEdgeDir);
						//edge_morton_vec_private.emplace_back(std::make_pair(p1 + back_t * modelEdgeDir, nodeMorton));
					}
				}
			}
		}

#pragma omp critical
		{
			edgeInterPoints.insert(edgeInterPoints.end(), edge_vec_private.begin(), edge_vec_private.end());
		}
	}

	std::sort(edgeInterPoints.begin(), edgeInterPoints.end(), std::less<V3d>());
	edgeInterPoints.erase(std::unique(edgeInterPoints.begin(), edgeInterPoints.end()), edgeInterPoints.end());
	allInterPoints.insert(allInterPoints.end(), edgeInterPoints.begin(), edgeInterPoints.end());

	stopTimer(&timer);
	double time = getElapsedTime(&timer) * 1e-3;
	test_time::test_allTime += time;

	cout << "-- 三角形边与node的交点数量：" << edgeInterPoints.size() << endl;

	// 三角形面与node边线交
	std::cout << "2. Computing the intersections between mesh FACES and node EDGES..." << endl;

	resetTimer(&timer);

	startTimer(&timer);

	const int numFineNodeEdges = fineNodeEdges.size();
	vector<node_edge_type> t_fineNodeEdges(numFineNodeEdges);
	std::transform(fineNodeEdges.begin(), fineNodeEdges.end(), t_fineNodeEdges.begin(), [](const node_edge_type& a) {
		if (!isLess<V3d>(a.first.first, a.first.second, std::less<V3d>())) {
			node_edge_type _p;
			_p.first.first = a.first.second, _p.first.second = a.first.first;
			_p.second = a.second;
			return _p;
		}
		else return a;
		});

	// 按末端点的x坐标从小到大排序
	struct x_sortEdge
	{
		bool operator()(node_edge_type& a, node_edge_type& b) {
			if (fabs(a.first.second.x() - b.first.second.x()) < 1e-9) // 若x坐标相等(剩下两个轴哪个先哪个后无所谓了)
			{
				if (fabs(a.first.second.y() - b.first.second.y()) < 1e-9)  // 若x和y坐标都相等
					return a.first.second.z() < b.first.second.z(); // 返回z小的那个
				else
					return a.first.second.y() < b.first.second.y(); // 返回y小的那个
			}
			else return a.first.second.x() < b.first.second.x();
		}
	};
	// 按末端点的y坐标从小到大排序
	struct y_sortEdge
	{
		bool operator()(node_edge_type& a, node_edge_type& b) {
			if (fabs(a.first.second.y() - b.first.second.y()) < 1e-9) // 若y坐标相等
			{
				if (fabs(a.first.second.x() - b.first.second.x()) < 1e-9)  // 若y和x坐标都相等
					return a.first.second.z() < b.first.second.z(); // 返回z小的那个
				else
					return a.first.second.x() < b.first.second.x(); // 返回x小的那个
			}
			else return a.first.second.y() < b.first.second.y();
		}
	};
	// 按末端点的z坐标从小到大排序
	struct z_sortEdge
	{
		bool operator()(node_edge_type& a, node_edge_type& b) {
			if (fabs(a.first.second.z() - b.first.second.z()) < 1e-9) // 若z坐标相等
			{
				if (fabs(a.first.second.x() - b.first.second.x()) < 1e-9)  // 若z和x坐标都相等
					return a.first.second.y() < b.first.second.y(); // 返回y小的那个
				else
					return a.first.second.x() < b.first.second.x(); // 返回x小的那个
			}
			else return a.first.second.z() < b.first.second.z();
		}
	};

	std::vector<node_edge_type> x_fineNodeEdges;
	std::vector<node_edge_type> y_fineNodeEdges;
	std::vector<node_edge_type> z_fineNodeEdges;
	//#pragma omp parallel
	{
		std::copy_if(t_fineNodeEdges.begin(), t_fineNodeEdges.end(), std::back_inserter(x_fineNodeEdges),
			[](const node_edge_type& val) { return parallelAxis(val.first.first, val.first.second) == 1; });
		std::sort(x_fineNodeEdges.begin(), x_fineNodeEdges.end(), x_sortEdge());

		std::copy_if(t_fineNodeEdges.begin(), t_fineNodeEdges.end(), std::back_inserter(y_fineNodeEdges),
			[](const node_edge_type& val) { return parallelAxis(val.first.first, val.first.second) == 2; });
		std::sort(y_fineNodeEdges.begin(), y_fineNodeEdges.end(), y_sortEdge());

		std::copy_if(t_fineNodeEdges.begin(), t_fineNodeEdges.end(), std::back_inserter(z_fineNodeEdges),
			[](const node_edge_type& val) { return parallelAxis(val.first.first, val.first.second) == 3; });
		std::sort(z_fineNodeEdges.begin(), z_fineNodeEdges.end(), z_sortEdge());
	}

	struct lessXVal {
		bool operator()(const node_edge_type& a, const node_edge_type& b) { // Search for first element 'a' in list such that b ≤ a
			return isLargeDouble(b.first.second.x(), a.first.second.x(), 1e-9);
		}
	};
	struct lessYVal {
		bool operator()(const node_edge_type& a, const node_edge_type& b) { // Search for first element 'a' in list such that b ≤ a
			return isLargeDouble(b.first.second.y(), a.first.second.y(), 1e-9);
		}
	};
	struct lessZVal {
		bool operator()(const node_edge_type& a, const node_edge_type& b) { // Search for first element 'a' in list such that b ≤ a
			return isLargeDouble(b.first.second.z(), a.first.second.z(), 1e-9);
		}
	};

	const size_t x_numEdges = x_fineNodeEdges.size();
	const size_t y_numEdges = y_fineNodeEdges.size();
	const size_t z_numEdges = z_fineNodeEdges.size();

#pragma omp parallel
	for (const auto& tri : modelTris)
	{
		V3d triEdge_1 = tri.p2 - tri.p1; V3d triEdge_2 = tri.p3 - tri.p2; V3d triEdge_3 = tri.p1 - tri.p3;
		V3d triNormal = tri.normal; double triDir = tri.dir;
		V3d tri_bbox_origin = fminf(tri.p1, fminf(tri.p2, tri.p3));
		V3d tri_bbox_end = fmaxf(tri.p1, fmaxf(tri.p2, tri.p3));

		// Search for first element x such that _q ≤ x
		node_edge_type x_q; x_q.first.second = Eigen::Vector3d(tri_bbox_origin.x(), 0, 0);
		auto x_lower = std::lower_bound(x_fineNodeEdges.begin(), x_fineNodeEdges.end(), x_q, lessXVal());;
		if (x_lower != x_fineNodeEdges.end())
		{
			std::vector<V3d> x_face_vec_private;

#pragma omp for nowait
			for (int i = x_lower - x_fineNodeEdges.begin(); i < x_numEdges; ++i)
			{
				auto e_p1 = x_fineNodeEdges[i].first.first, e_p2 = x_fineNodeEdges[i].first.second;

				if (isLargeDouble(e_p1.x(), tri_bbox_end.x(), 1e-9)) break; // 起始端点大于bbox_end

				V3d edgeDir = e_p2 - e_p1;

				if (fabsf(triNormal.dot(edgeDir)) < 1e-9) continue;

				double t = (-triDir - triNormal.dot(e_p1)) / (triNormal.dot(edgeDir));
				if (t < 0. || t > 1.) continue;
				V3d interPoint = e_p1 + edgeDir * t;

				if (triEdge_1.cross(interPoint - tri.p1).dot(triNormal) < 0) continue;
				if (triEdge_2.cross(interPoint - tri.p2).dot(triNormal) < 0) continue;
				if (triEdge_3.cross(interPoint - tri.p3).dot(triNormal) < 0) continue;

				x_face_vec_private.emplace_back(interPoint);
			}
			if (!x_face_vec_private.empty())
			{
#pragma omp critical
				{
					faceInterPoints.insert(faceInterPoints.end(), x_face_vec_private.begin(), x_face_vec_private.end());
				}
			}
		}

		node_edge_type y_q; y_q.first.second = Eigen::Vector3d(0, tri_bbox_origin.y(), 0);
		// Search for first element x such that _q ≤ x
		auto y_lower = std::lower_bound(y_fineNodeEdges.begin(), y_fineNodeEdges.end(), y_q, lessYVal());
		if (y_lower != y_fineNodeEdges.end())
		{
			std::vector<V3d> y_face_vec_private;

#pragma omp for nowait
			for (int i = y_lower - y_fineNodeEdges.begin(); i < y_numEdges; ++i)
			{
				auto e_p1 = y_fineNodeEdges[i].first.first, e_p2 = y_fineNodeEdges[i].first.second;

				if (isLargeDouble(e_p1.y(), tri_bbox_end.y(), 1e-9)) break; // 起始端点大于bbox_end

				V3d edgeDir = e_p2 - e_p1;

				if (fabsf(triNormal.dot(edgeDir)) < 1e-9) continue;

				double t = (-triDir - triNormal.dot(e_p1)) / (triNormal.dot(edgeDir));
				if (t < 0. || t > 1.) continue;
				V3d interPoint = e_p1 + edgeDir * t;

				if (triEdge_1.cross(interPoint - tri.p1).dot(triNormal) < 0) continue;
				if (triEdge_2.cross(interPoint - tri.p2).dot(triNormal) < 0) continue;
				if (triEdge_3.cross(interPoint - tri.p3).dot(triNormal) < 0) continue;

				y_face_vec_private.emplace_back(interPoint);
			}
			if (!y_face_vec_private.empty())
			{
#pragma omp critical
				{
					faceInterPoints.insert(faceInterPoints.end(), y_face_vec_private.begin(), y_face_vec_private.end());
				}
			}
		}

		// Search for first element x such that _q ≤ x
		node_edge_type z_q; z_q.first.second = Eigen::Vector3d(0, 0, tri_bbox_origin.z());
		auto z_lower = std::lower_bound(z_fineNodeEdges.begin(), z_fineNodeEdges.end(), z_q, lessZVal());
		if (z_lower != z_fineNodeEdges.end())
		{
			std::vector<V3d> z_face_vec_private;

#pragma omp for nowait
			for (int i = z_lower - z_fineNodeEdges.begin(); i < z_numEdges; ++i)
			{
				auto e_p1 = z_fineNodeEdges[i].first.first, e_p2 = z_fineNodeEdges[i].first.second;

				if (isLargeDouble(e_p1.z(), tri_bbox_end.z(), 1e-9)) break; // 起始端点大于bbox_end

				V3d edgeDir = e_p2 - e_p1;

				if (fabsf(triNormal.dot(edgeDir)) < 1e-9) continue;

				double t = (-triDir - triNormal.dot(e_p1)) / (triNormal.dot(edgeDir));
				if (t < 0. || t > 1.) continue;
				V3d interPoint = e_p1 + edgeDir * t;

				if (triEdge_1.cross(interPoint - tri.p1).dot(triNormal) < 0) continue;
				if (triEdge_2.cross(interPoint - tri.p2).dot(triNormal) < 0) continue;
				if (triEdge_3.cross(interPoint - tri.p3).dot(triNormal) < 0) continue;

				z_face_vec_private.emplace_back(interPoint);
			}
			if (!z_face_vec_private.empty())
			{
#pragma omp critical
				{
					faceInterPoints.insert(faceInterPoints.end(), z_face_vec_private.begin(), z_face_vec_private.end());
				}
			}
		}
	}

	//#pragma omp parallel
	//	for (const auto& tri : modelTris)
	//	{
	//		std::vector<V3d> face_vec_private;
	//		V3d triEdge_1 = tri.p2 - tri.p1; V3d triEdge_2 = tri.p3 - tri.p2; V3d triEdge_3 = tri.p1 - tri.p3;
	//		V3d triNormal = tri.normal; double triDir = tri.dir;
	//		V3d tri_bbox_origin = fminf(tri.p1, fminf(tri.p2, tri.p3));
	//		V3d tri_bbox_end = fmaxf(tri.p1, fmaxf(tri.p2, tri.p3));
	//
	//		double d_eps = 1e-9;
	//#pragma omp for nowait
	//		for (int j = 0; j < numFineNodeEdges; ++j)
	//		{
	//			auto nodeEdge = fineNodeEdges[j];
	//
	//			auto e_p1 = nodeEdge.first.first, e_p2 = nodeEdge.first.second;
	//			if (!isLess(e_p1, e_p2, std::less<V3d>())) std::swap(e_p1, e_p2);
	//
	//			const int parallel = parallelAxis(e_p1, e_p2);
	//			if (parallel == 1) // 与x轴平行(两端点y和z值相等)
	//			{
	//				if (isLessDouble(e_p1.y(), tri_bbox_origin.y(), d_eps) || isLargeDouble(e_p1.y(), tri_bbox_end.y(), d_eps) ||
	//					isLessDouble(e_p1.z(), tri_bbox_origin.z(), d_eps) || isLargeDouble(e_p1.z(), tri_bbox_end.z(), d_eps) ||
	//					isLessDouble(e_p2.x(), tri_bbox_origin.x(), d_eps) || isLargeDouble(e_p1.x(), tri_bbox_end.x(), d_eps)) continue;
	//			}
	//			else if (parallel == 2) // 与y轴平行
	//			{
	//				if (isLessDouble(e_p1.x(), tri_bbox_origin.x(), d_eps) || isLargeDouble(e_p1.x(), tri_bbox_end.x(), d_eps) ||
	//					isLessDouble(e_p1.z(), tri_bbox_origin.z(), d_eps) || isLargeDouble(e_p1.z(), tri_bbox_end.z(), d_eps) ||
	//					isLessDouble(e_p2.y(), tri_bbox_origin.y(), d_eps) || isLargeDouble(e_p1.y(), tri_bbox_end.y(), d_eps)) continue;
	//			}
	//			else // 与z轴平行
	//			{
	//				if (isLessDouble(e_p1.x(), tri_bbox_origin.x(), d_eps) || isLargeDouble(e_p1.x(), tri_bbox_end.x(), d_eps) ||
	//					isLessDouble(e_p1.y(), tri_bbox_origin.y(), d_eps) || isLargeDouble(e_p1.y(), tri_bbox_end.y(), d_eps) ||
	//					isLessDouble(e_p2.z(), tri_bbox_origin.z(), d_eps) || isLargeDouble(e_p1.z(), tri_bbox_end.z(), d_eps)) continue;
	//			}
	//
	//			V3d edgeDir = e_p2 - e_p1;
	//
	//			if (fabsf(triNormal.dot(edgeDir)) < 1e-9) continue;
	//
	//			double t = (-triDir - triNormal.dot(e_p1)) / (triNormal.dot(edgeDir));
	//			if (t < 0. || t > 1.) continue;
	//			V3d interPoint = e_p1 + edgeDir * t;
	//
	//			if (triEdge_1.cross(interPoint - tri.p1).dot(triNormal) < 0) continue;
	//			if (triEdge_2.cross(interPoint - tri.p2).dot(triNormal) < 0) continue;
	//			if (triEdge_3.cross(interPoint - tri.p3).dot(triNormal) < 0) continue;
	//
	//			face_vec_private.emplace_back(interPoint);
	//		}
	//
	//#pragma omp critical
	//		{
	//			faceInterPoints.insert(faceInterPoints.end(), face_vec_private.begin(), face_vec_private.end());
	//		}
	//	}

		//#pragma omp parallel
		//	for (const auto& tri : modelTris)
		//	{
		//		std::vector<V3d> face_vec_private;
		//
		//		V3d triEdge_1 = tri.p2 - tri.p1; V3d triEdge_2 = tri.p3 - tri.p2; V3d triEdge_3 = tri.p1 - tri.p3;
		//		V3d triNormal = tri.normal; double triDir = tri.dir;
		//
		//#pragma omp for nowait
		//		for (int j = 0; j < numFineNodeEdges; ++j)
		//		{
		//			const auto& nodeEdge = fineNodeEdges[j];
		//			thrust_edge_type edge = nodeEdge.first;
		//			V3d edgeDir = edge.second - edge.first;
		//
		//			if (fabsf(triNormal.dot(edgeDir)) < 1e-9) continue;
		//
		//			double t = (-triDir - triNormal.dot(edge.first)) / (triNormal.dot(edgeDir));
		//			if (t < 0. || t > 1.) continue;
		//			V3d interPoint = edge.first + edgeDir * t;
		//
		//			if (triEdge_1.cross(interPoint - tri.p1).dot(triNormal) < 0) continue;
		//			if (triEdge_2.cross(interPoint - tri.p2).dot(triNormal) < 0) continue;
		//			if (triEdge_3.cross(interPoint - tri.p3).dot(triNormal) < 0) continue;
		//
		//			//faceInterPoints.emplace_back(interPoint);
		//			face_vec_private.emplace_back(interPoint);
		//		}
		//
		//#pragma omp critical
		//		{
		//			faceInterPoints.insert(faceInterPoints.end(), face_vec_private.begin(), face_vec_private.end());
		//		}
		//	}

	allInterPoints.insert(allInterPoints.end(), faceInterPoints.begin(), faceInterPoints.end());

	stopTimer(&timer);
	time = getElapsedTime(&timer) * 1e-3;
	test_time::test_allTime += time;

	cout << "-- 三角形面与node边的交点数量：" << faceInterPoints.size() << endl;
	cout << "-- 总交点数量：" << allInterPoints.size() << endl;

	deleteTimer(&timer);

	saveLatentPoint("");
}

inline void ThinShells::cpSDFOfTreeNodes()
{
	TimerInterface* timer = nullptr;
	createTimer(&timer);
	startTimer(&timer);

	const auto& depthNodeVertexArray = svo.depthNodeVertexArray;
	const auto& esumDepthNodeVerts = svo.esumDepthNodeVerts;
	const size_t& numNodeVerts = svo.numNodeVerts;
	MXd pointsMat(numNodeVerts, 3);
	for (int d = 0; d < treeDepth; ++d)
	{
		const size_t& d_numNodeVerts = depthNodeVertexArray[d].size();
		for (int i = 0; i < d_numNodeVerts; ++i)
			pointsMat.row(esumDepthNodeVerts[d] + i) = depthNodeVertexArray[d][i].first;
	}

	{
		VXi I;
		MXd C, N;
		igl::signed_distance(pointsMat, m_V, m_F, igl::SignedDistanceType::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER, sdfVal, I, C, N);
	}

	stopTimer(&timer);
	double time = getElapsedTime(&timer) * 1e-3;
	test_time::test_allTime += time;

	deleteTimer(&timer);
}

inline void ThinShells::cpCoefficients()
{
	using SpMat = Eigen::SparseMatrix<double>;
	using Trip = Eigen::Triplet<double>;

	const auto& nodeArray = svo.svoNodeArray;
	const vector<vector<node_vertex_type>>& depthNodeVertexArray = svo.depthNodeVertexArray;
	const vector<size_t>& esumDepthNodeVerts = svo.esumDepthNodeVerts;
	depthVert2Idx.resize(treeDepth);

	TimerInterface* timer = nullptr;
	createTimer(&timer);
	startTimer(&timer);

	// 每层顶点到顶点下标(全局)的映射
	for (int d = 0; d < treeDepth; ++d)
	{
		const size_t d_numNodeVerts = depthNodeVertexArray[d].size();
		vector<size_t> d_nodeVertexIdx(d_numNodeVerts);
		std::iota(d_nodeVertexIdx.begin(), d_nodeVertexIdx.end(), 0);

		const size_t& d_esumNodeVerts = esumDepthNodeVerts[d]; // 顶点数量的exclusive scan
		std::transform(depthNodeVertexArray[d].begin(), depthNodeVertexArray[d].end(),
			d_nodeVertexIdx.begin(), std::inserter(depthVert2Idx[d], depthVert2Idx[d].end()),
			[d_esumNodeVerts](const node_vertex_type& val, const size_t& idx) {
				return std::make_pair(val.first, d_esumNodeVerts + idx);
			});
	}

	/*for (int d = 0; d < treeDepth; ++d)
	{
		const size_t d_numNodeVerts = depthNodeVertexArray[d].size();
		vector<size_t> d_nodeVertexIdx(d_numNodeVerts);
		std::iota(d_nodeVertexIdx.begin(), d_nodeVertexIdx.end(), 0);

		std::transform(depthNodeVertexArray[d].begin(), depthNodeVertexArray[d].end(), d_nodeVertexIdx.begin(),
			std::inserter(depthVert2Idx[d], depthVert2Idx[d].end()),
			[](const node_vertex_type& val, size_t i) {
				return std::make_pair(val.first, i);
			});
	}*/
	// initial matrix
	const size_t& numNodeVerts = svo.numNodeVerts;
	vector<Trip> matApVal;

#pragma omp parallel
	for (int d = 0; d < treeDepth; ++d)
	{
		const size_t d_numNodeVerts = depthNodeVertexArray[d].size(); // 每层节点的顶点数量
		//const size_t& d_esumNodeVerts = esumDepthNodeVerts[d]; // 顶点数量的exclusive scan
#pragma omp for nowait
		for (int i = 0; i < d_numNodeVerts; ++i)
		{
			V3d i_nodeVertex = depthNodeVertexArray[d][i].first;
			uint32_t i_fromNodeIdx = depthNodeVertexArray[d][i].second;
			const size_t rowIdx = depthVert2Idx[d].at(i_nodeVertex);

			vector<Trip> private_matApVal;

			//matApVal.emplace_back(Trip(d_esumNodeVerts + i, d_esumNodeVerts + i, 1)); // self
			private_matApVal.emplace_back(Trip(rowIdx, rowIdx, 1)); // self

			//#pragma omp parallel for
			for (int j = d - 1; j >= 0; --j)
			{
				if (depthVert2Idx[j].find(i_nodeVertex) == depthVert2Idx[j].end()) break;
				//matApVal.emplace_back(Trip(d_esumNodeVerts + i, esumDepthNodeVerts[j] + nodeVertex2Idx[j][i_nodeVertex], 1)); // child
				private_matApVal.emplace_back(Trip(rowIdx, depthVert2Idx[j].at(i_nodeVertex), 1)); // child
			}

			// parent
			auto inDmPointTraits = svo.setInDomainPoints(nodeArray[i_fromNodeIdx].parent, d + 1, depthVert2Idx);
			//auto inDmPointTraits = svo.setInDomainPoints(i_fromNodeIdx, d + 1, depthVert2Idx);
			const int nInDmPoints = inDmPointTraits.size();

			//#pragma omp parallel for
			for (int k = 0; k < nInDmPoints; ++k)
			{
				const auto inDmPointTrait = inDmPointTraits[k];
				double val = BaseFunction4Point(std::get<0>(inDmPointTrait), std::get<1>(inDmPointTrait), i_nodeVertex);
				assert(inDmPointsIdx[k] < numNodeVerts, "index of col > numNodeVertex!");
				//if (val != 0) matApVal.emplace_back(Trip(d_esumNodeVerts + i, inDmPointsIdx[k], val));
				if (val != 0)  private_matApVal.emplace_back(Trip(rowIdx, std::get<2>(inDmPointTrait), val));
				//if (val != 0)  private_matApVal.emplace_back(Trip(d_esumNodeVerts + i, inDmPointsIdx[k], val));
			}

#pragma omp critical
			{
				matApVal.insert(matApVal.end(), private_matApVal.begin(), private_matApVal.end());
			}
		}
	}

	SpMat A(numNodeVerts, numNodeVerts); // Ap
	A.setFromTriplets(matApVal.begin(), matApVal.end());
	auto b = sdfVal;

	stopTimer(&timer);
	double time = getElapsedTime(&timer) * 1e-3;
	test_time::test_allTime += time;

	resetTimer(&timer);

	startTimer(&timer);
	Eigen::LeastSquaresConjugateGradient<SpMat> lscg;
	lscg.compute(A);
	lambda = lscg.solve(b);

	stopTimer(&timer);
	time = getElapsedTime(&timer) * 1e-3;
	test_time::test_allTime += time;

	printf("-- Solve equation elapsed time: %lf s.\n", time);
	deleteTimer(&timer);

	cout << "-- Residual Error: " << (A * lambda - b).norm() << endl;

	//saveCoefficients(concatFilePath((string)OUT_DIR, modelName, std::to_string(maxDepth), (string)"Coefficients.txt"));
	/*for (int u = 0; u < treeDepth; ++u)
	{
		const size_t u_numNodeVerts = depthNodeVertexArray[u].size();
		const size_t& u_esumNodeVerts = esumDepthNodeVerts[u];
		for (int k = 0; k < u_numNodeVerts; ++k)
		{
			double bSplineVal = 0, sdf = sdfVal[u_esumNodeVerts + k];
			V3d vert = depthNodeVertexArray[u][k].first;
			for (int d = 0; d < treeDepth; ++d)
			{
				const size_t d_numNodeVerts = depthNodeVertexArray[d].size();
				const size_t& d_esumNodeVerts = esumDepthNodeVerts[d];
				for (int j = 0; j < d_numNodeVerts; ++j)
				{
					V3d nodeVert = depthNodeVertexArray[d][j].first;
					uint32_t nodeIdx = depthNodeVertexArray[d][j].second;
					bSplineVal += lambda[d_esumNodeVerts + j] * (BaseFunction4Point(nodeVert, svo.svoNodeArray[nodeIdx].width, vert));
				}
			}

			if (fabs(bSplineVal - sdf) > 1e-9)
				printf("vertIdx = %llu, bSplineVal = %.10lf, sdf = %.10lf\n", u_esumNodeVerts + k, bSplineVal, sdf);
		}
	}*/
}

inline void ThinShells::cpLatentBSplineValue()
{
	TimerInterface* timer = nullptr;
	createTimer(&timer);
	startTimer(&timer);

	const uint numAllPoints = nModelVerts + allInterPoints.size();
	std::vector<V3d> pointsData;
	pointsData.insert(pointsData.end(), modelVerts.begin(), modelVerts.end());
	pointsData.insert(pointsData.end(), allInterPoints.begin(), allInterPoints.end());
	const auto& svoNodeArray = svo.svoNodeArray;

	if (nodeWidthArray.empty())
	{
		std::transform(svoNodeArray.begin(), svoNodeArray.end(), std::back_inserter(nodeWidthArray),
			[](SVONode node) {
				return Eigen::Vector3d(node.width, node.width, node.width);
			});
	}

	const uint nInterPoints = allInterPoints.size();
	bSplineVal.resize(numAllPoints);
	bSplineVal.setZero();

	/*cuAcc::cpBSplineVal(numAllPoints, svo.numNodeVerts, svo.numTreeNodes, pointsData,
		svo.nodeVertexArray, nodeWidthArray, lambda, bSplineVal);*/

#pragma omp parallel
	for (size_t i = 0; i < numAllPoints; ++i)
	{
		const V3d& point = pointsData[i];

		double sum = 0.0;
		V3i dis = getPointDis(point, modelOrigin, voxelWidth);
		const uint32_t pointMorton = morton::mortonEncode_LUT((uint16_t)dis.x(), (uint16_t)dis.y(), (uint16_t)dis.z());
		const uint32_t nodeIdx = morton2FineNodeIdx.at(pointMorton);

		auto inDmPointsTraits = svo.setInDomainPoints(nodeIdx, 0, depthVert2Idx);
		const int nInDmPointsTraits = inDmPointsTraits.size();

#pragma omp parallel for reduction(+ : sum) // for循环中的变量必须得是有符号整型
		for (int j = 0; j < nInDmPointsTraits; ++j)
		{
			const auto& inDmPointTrait = inDmPointsTraits[j];
			sum += lambda[std::get<2>(inDmPointTrait)] * BaseFunction4Point(std::get<0>(inDmPointTrait), std::get<1>(inDmPointTrait), point);
		}
		bSplineVal[i] = sum;
	}

	// --CPU--
	/*const vector<SVONode>& svoNodeArray = svo.svoNodeArray;
	const vector<vector<node_vertex_type>>& depthNodeVertexArray = svo.depthNodeVertexArray;
	const vector<size_t>& esumDepthNodeVerts = svo.esumDepthNodeVerts;

	for (int i = 0; i < nModelVerts; ++i)
	{
		const V3d& modelVert = modelVerts[i];
		for (int d = 0; d < treeDepth; ++d)
		{
			const size_t d_numNodeVerts = depthNodeVertexArray[d].size();
			const size_t& d_esumNodeVerts = esumDepthNodeVerts[d];
			for (int j = 0; j < d_numNodeVerts; ++j)
			{
				V3d nodeVert = depthNodeVertexArray[d][j].first;
				uint32_t nodeIdx = depthNodeVertexArray[d][j].second;
				bSplineVal[i] += lambda[d_esumNodeVerts + j] * (BaseFunction4Point(nodeVert, svoNodeArray[nodeIdx].width, modelVert));
			}
		}
	}

	int cnt = 0;
	for (int i = 0; i < nInterPoints; ++i)
	{
		cnt = i + nModelVerts;
		const V3d& interPoint = allInterPoints[i];
		for (int d = 0; d < treeDepth; ++d)
		{
			const size_t d_numNodeVerts = depthNodeVertexArray[d].size();
			const size_t& d_esumNodeVerts = esumDepthNodeVerts[d];
			for (int j = 0; j < d_numNodeVerts; ++j)
			{
				V3d nodeVert = depthNodeVertexArray[d][j].first;
				uint32_t nodeIdx = depthNodeVertexArray[d][j].second;
				bSplineVal[cnt] += lambda[d_esumNodeVerts + j] * (BaseFunction4Point(nodeVert, svoNodeArray[nodeIdx].width, interPoint));
			}
		}
	}*/

	innerShellIsoVal = *(std::min_element(bSplineVal.begin(), bSplineVal.end()));
	outerShellIsoVal = *(std::max_element(bSplineVal.begin(), bSplineVal.end()));

	stopTimer(&timer);
	double time = getElapsedTime(&timer) * 1e-3;
	test_time::test_allTime += time;

	deleteTimer(&timer);

	std::cout << "-- innerShellIsoVal: " << innerShellIsoVal << std::endl;
	std::cout << "-- outerShellIsoVal: " << outerShellIsoVal << std::endl;
}

inline void ThinShells::initBSplineTree()
{
	TimerInterface* timer = nullptr;
	createTimer(&timer);

	cout << "\nComputing intersection points of " << std::quoted(modelName) << "and level-0 nodes...\n=====================" << endl;
	startTimer(&timer);
	cpIntersectionPoints();
	stopTimer(&timer);
	double time = getElapsedTime(&timer) * 1e-3;
	//qp::qp_ctrl(tColor::GREEN);
	//qp::qprint("-- Elapsed time: ", time, "s.");
	printf("-- Elapsed time: %lf s.\n", time);
	//qp::qp_ctrl();
	cout << "=====================\n";
#ifdef IO_SAVE
	saveIntersections("", "");
#endif // IO_SAVE

	cout << "\nComputing discrete SDF of tree nodes..." << endl;
	startTimer(&timer);
	cpSDFOfTreeNodes();
	stopTimer(&timer);
	time = getElapsedTime(&timer) * 1e-3;
	printf("-- Elapsed time: %lf s.\n", time);
	cout << "=====================\n";
#ifdef IO_SAVE
	saveSDFValue("");
#endif // IO_SAVE

	cout << "\nComputing coefficients..." << endl;
	startTimer(&timer);
	cpCoefficients();
	stopTimer(&timer);
	time = getElapsedTime(&timer) * 1e-3;
	printf("-- Elapsed time: %lf s.\n", time);
	cout << "=====================\n";
#ifdef IO_SAVE
	saveCoefficients("");
#endif // IO_SAVE

	cout << "\nComputing B-Spline value..." << endl;
	startTimer(&timer);
	cpLatentBSplineValue();
	stopTimer(&timer);
	time = getElapsedTime(&timer) * 1e-3;
	printf("-- Elapsed time: %lf s.\n", time);
	cout << "=====================\n";
#ifdef IO_SAVE
	saveBSplineValue("");
#endif // IO_SAVE

	deleteTimer(&timer);
}

void ThinShells::creatShell()
{
	initBSplineTree();
}

inline void ThinShells::setLatentMatrix(const double& alpha)
{
	//using SpMat = Eigen::SparseMatrix<double>;
	//using Trip = Eigen::Triplet<double>;
	//
	//const vector<vector<node_vertex_type>>& depthNodeVertexArray = svo.depthNodeVertexArray;
	//const vector<size_t>& esumDepthNodeVerts = svo.esumDepthNodeVerts;
	//vector<std::map<V3d, size_t>> nodeVertex2Idx(treeDepth);
	//const size_t& numNodeVerts = svo.numNodeVerts;
	//
	//vector<Trip> matAlVal;
	//const uint nInterPoints = allInterPoints.size();
	//const vector<SVONode>& nodeArray = svo.svoNodeArray;
	//for (int i = 0; i < nInterPoints; ++i)
	//{
	//	V3d interPoint = allInterPoints[i].first;
	//	uint32_t fromNodeIdx = allInterPoints[i].second;
	//	// 自身所处格子的八个顶点和所有父节点的八个顶点
	//	auto [inDmPoints, inDmPointsIdx] = svo.setInDomainPoints(fromNodeIdx, 0, esumDepthNodeVerts, nodeVertex2Idx);
	//	const int nInDmPoints = inDmPoints.size();
	//
	//	for (int k = 0; k < nInDmPoints; ++k)
	//	{
	//		double val = BaseFunction4Point(inDmPoints[k].first, inDmPoints[k].second, interPoint);
	//		if (val != 0) matAlVal.emplace_back(Trip(numNodeVerts + i, inDmPointsIdx[k], alpha * val));
	//	}
	//}
	//
	//const V3i& surfaceVoxelGridSize = svo.surfaceVoxelGridSize;
	//const V3i grid_max = V3i(surfaceVoxelGridSize.x() - 1, surfaceVoxelGridSize.y() - 1, surfaceVoxelGridSize.z() - 1);
	//const V3d unitVoxelSize = V3d(modelBoundingBox.boxWidth.x() / surfaceVoxelGridSize.x(),
	//	modelBoundingBox.boxWidth.y() / surfaceVoxelGridSize.y(),
	//	modelBoundingBox.boxWidth.z() / surfaceVoxelGridSize.z());
	//std::map<uint32_t, uint32_t> morton2Idx;
	//const size_t& numFineNodes = svo.numFineNodes;
	//vector<uint32_t> nodeIdx(numFineNodes);
	//std::iota(nodeIdx.begin(), nodeIdx.end(), 0);
	//std::transform(nodeArray.begin(), nodeArray.begin() + numFineNodes, nodeIdx.begin(),
	//	std::inserter(morton2Idx, morton2Idx.end()),
	//	[](const SVONode& val, uint32_t i) {
	//		return std::make_pair(val.mortonCode, i);
	//	});
	//
	//auto getFromNodeIdx = [&](const V3d& modelVertex)->uint32_t
	//{
	//	const V3d gridVertex = modelVertex - modelBoundingBox.boxOrigin;
	//	const V3i gridIdx = clamp(
	//		V3i((gridVertex.x() / unitVoxelSize.x()), (gridVertex.y() / unitVoxelSize.y()), (gridVertex.z() / unitVoxelSize.z())),
	//		V3i(0, 0, 0), grid_max
	//	);
	//	return morton2Idx[morton::mortonEncode_LUT((uint16_t)gridIdx.x(), (uint16_t)gridIdx.y(), (uint16_t)gridIdx.z())];
	//};
	//
	//for (int i = 0; i < nModelVerts; ++i)
	//{
	//	V3d modelVertex = modelVerts[i];
	//	uint32_t i_fromNodeIdx = getFromNodeIdx(modelVertex);
	//	// 自身所处格子的八个顶点和所有父节点的八个顶点
	//	auto [inDmPoints, inDmPointsIdx] = svo.setInDomainPoints(i_fromNodeIdx, 0, esumDepthNodeVerts, nodeVertex2Idx);
	//	const int nInDmPoints = inDmPoints.size();
	//
	//	for (int k = 0; k < nInDmPoints; ++k)
	//	{
	//		double val = BaseFunction4Point(inDmPoints[k].first, inDmPoints[k].second, modelVertex);
	//		if (val != 0) matApVal.emplace_back(Trip(numNodeVerts + nInterPoints + i, inDmPointsIdx[k], alpha * val));
	//	}
	//}
	//
	////SpMat sm_Ap(numNodeVerts, numNodeVerts); // Ap
	//SpMat sm_Ap(numNodeVerts + nInterPoints + nModelVerts, numNodeVerts); // Ap
	//sm_Ap.setFromTriplets(matApVal.begin(), matApVal.end());
	//SpMat sm_Al(nInterPoints + nModelVerts, numNodeVerts); // Al
	//sm_Al.setFromTriplets(matAlVal.begin(), matAlVal.end());
	//auto Ap = /*alpha * sm_Ap.transpose() * */sm_Ap;
	//auto Al = sm_Al.transpose() * sm_Al;
	//auto A = Ap/* + Al*/;
	//auto b = /*alpha * sm_Ap.transpose() * */sdfVal;
	//
	////Eigen::SimplicialLLT<SpMat> sllt;
	//Eigen::LeastSquaresConjugateGradient<SpMat> lscg;
	//lscg.compute(A);
	//lambda = lscg.solve(b);
}

void ThinShells::singlePointQuery(const std::string& out_file, const V3d& point)
{
	if (innerShellIsoVal == -DINF || outerShellIsoVal == -DINF) { printf("Error: You must create shells first!"); return; }
	if (nodeWidthArray.empty())
	{
		auto& svoNodeArray = svo.svoNodeArray;
		std::transform(svoNodeArray.begin(), svoNodeArray.end(), std::back_inserter(nodeWidthArray),
			[](SVONode node) {
				return Eigen::Vector3d(node.width, node.width, node.width);
			});
	}

	double q_bSplineVal;
	V3d rgb;
	cuAcc::cpBSplineVal(svo.numNodeVerts, svo.numTreeNodes, point, svo.nodeVertexArray, nodeWidthArray, lambda, q_bSplineVal);
	if (innerShellIsoVal < q_bSplineVal && q_bSplineVal < outerShellIsoVal) rgb = V3d(0.56471, 0.93333, 0.56471);
	else rgb = V3d(1, 0.27059, 0);

	string _out_file = out_file;
	if (getFileExtension(_out_file) != ".obj")
		_out_file = (string)getDirName(DELIMITER, out_file.c_str()) + (string)getFileName(DELIMITER, out_file.c_str()) + (string)".obj";

	checkDir(_out_file);
	std::ofstream out(_out_file);
	if (!out) { fprintf(stderr, "[I/O] Error: File %s could not be opened!", _out_file.c_str()); return; }
	cout << "-- Save query result to " << std::quoted(_out_file) <<
		"-- [RED] point not on the surface, [GREEN] point lie on the surface" << endl;

	gvis::writePointCloud(point, rgb, out);
}

void ThinShells::multiPointQuery(const std::string& out_file, const vector<V3d>& points)
{
	if (innerShellIsoVal == -DINF || outerShellIsoVal == -DINF) { printf("Error: You must create shells first!"); return; }
	if (nodeWidthArray.empty())
	{
		auto& svoNodeArray = svo.svoNodeArray;
		std::transform(svoNodeArray.begin(), svoNodeArray.end(), std::back_inserter(nodeWidthArray),
			[](SVONode node) {
				return Eigen::Vector3d(node.width, node.width, node.width);
			});
	}

	VXd q_bSplineVal; vector<V3d> rgbs;
	cuAcc::cpBSplineVal(points.size(), svo.numNodeVerts, svo.numTreeNodes,
		points, svo.nodeVertexArray, nodeWidthArray, lambda, q_bSplineVal);
	std::transform(q_bSplineVal.begin(), q_bSplineVal.end(), std::back_inserter(rgbs),
		[=](double val) {
			V3d _t;
			if (innerShellIsoVal < val && val < outerShellIsoVal) _t = V3d(0.56471, 0.93333, 0.56471);
			else _t = V3d(1, 0.27059, 0);
			return _t;
		});

	string _out_file = out_file;
	if (getFileExtension(_out_file) != ".obj")
		_out_file = (string)getDirName(DELIMITER, out_file.c_str()) + (string)getFileName(DELIMITER, out_file.c_str()) + (string)".obj";

	checkDir(_out_file);
	std::ofstream out(_out_file);
	if (!out) { fprintf(stderr, "[I/O] Error: File %s could not be opened!", _out_file.c_str()); return; }
	cout << "-- Save query result to " << std::quoted(_out_file) <<
		"-- [RED] points not on the surface, [GREEN] points lie on the surface" << endl;

	gvis::writePointCloud(points, rgbs, out);
}

void ThinShells::multiPointQuery(const std::string& out_file, const MXd& pointsMat)
{
	if (innerShellIsoVal == -DINF || outerShellIsoVal == -DINF) { printf("Error: You must create shells first!"); return; }

	vector<V3d> points;
	for (int i = 0; i < pointsMat.rows(); ++i) points.emplace_back(pointsMat.row(i));

	multiPointQuery(out_file, points);
}

//////////////////////
//  I/O: Save Data  //
//////////////////////
void ThinShells::saveTree(const string& filename) const
{
	string t_filename = filename;
	//if (filename.empty()) t_filename = concatFilePath((string)VIS_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"svo.obj");
	if (filename.empty()) t_filename = concatFilePath((string)VIS_DIR, modelName, uniformDir, std::to_string(treeDepth));

	svo.saveSVO(t_filename);
}

void ThinShells::saveIntersections(const string& filename, const vector<V3d>& intersections) const
{
	checkDir(filename);
	std::ofstream out(filename);
	if (!out) { fprintf(stderr, "[I/O] Error: File %s could not be opened!", filename.c_str()); return; }

	for (const auto& p : intersections)
		out << p.x() << " " << p.y() << " " << p.z() << endl;
	out.close();
}

void ThinShells::saveIntersections(const string& filename_1, const string& filename_2) const
{
	string t_filename = filename_1;
	if (filename_1.empty())
		t_filename = concatFilePath((string)VIS_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"edgeInter.xyz");
	cout << "-- Save mesh EDGES and octree Nodes to " << std::quoted(t_filename) << endl;
	saveIntersections(t_filename, edgeInterPoints);

	t_filename = filename_2;
	if (filename_2.empty())
		t_filename = concatFilePath((string)VIS_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"faceInter.xyz");
	cout << "-- Save mesh FACES and octree node EDGES to " << std::quoted(t_filename) << endl;
	saveIntersections(t_filename, faceInterPoints);
}

void ThinShells::saveSDFValue(const string& filename) const
{
	string t_filename = filename;
	if (filename.empty())
		t_filename = concatFilePath((string)OUT_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"SDFValue.txt");

	checkDir(t_filename);
	std::ofstream out(t_filename);
	if (!out) { fprintf(stderr, "[I/O] Error: File %s could not open!\n", filename.c_str()); return; }

	cout << "-- Save SDF value to " << std::quoted(t_filename) << endl;
	for (const auto& val : sdfVal)
		out << val << endl;
	out.close();
}

void ThinShells::saveCoefficients(const string& filename) const
{
	string t_filename = filename;
	if (filename.empty())
		t_filename = concatFilePath((string)OUT_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"Coefficients.txt");

	checkDir(t_filename);
	std::ofstream out(t_filename);
	if (!out) { fprintf(stderr, "[I/O] Error: File %s could not open!\n", t_filename.c_str()); return; }

	cout << "-- Save coefficients to " << std::quoted(t_filename) << endl;
	for (const auto& val : lambda)
		out << val << endl;
}

void ThinShells::saveLatentPoint(const string& filename) const
{
	string t_filename = filename;
	if (filename.empty())
		t_filename = concatFilePath((string)OUT_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"latent_point.xyz");

	checkDir(t_filename);
	//std::ofstream out(t_filename);
	//std::ofstream out(t_filename, std::ios_base::app);
	std::ofstream out(t_filename, std::ofstream::out | std::ofstream::trunc);
	if (!out) { fprintf(stderr, "[I/O] Error: File %s could not open!\n", t_filename.c_str()); return; }

	cout << "-- Save latent point to " << std::quoted(t_filename) << endl;

	gvis::writePointCloud_xyz(m_V, out);
	out.close(); out.open(t_filename, std::ofstream::app);
	gvis::writePointCloud_xyz(allInterPoints, out);
	out.close();
}

void ThinShells::saveBSplineValue(const string& filename) const
{
	string t_filename = filename;
	if (filename.empty())
		t_filename = concatFilePath((string)OUT_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"BSplineValue.txt");

	checkDir(t_filename);
	std::ofstream out(t_filename);
	if (!out) { fprintf(stderr, "[I/O] Error: File %s could not open!\n", t_filename.c_str()); return; }

	cout << "-- Save B-Spline value to " << std::quoted(t_filename) << endl;
	out << std::setiosflags(std::ios::fixed) << std::setprecision(9) << bSplineVal << endl;
	out.close();
}

//////////////////////
//   Visualiztion   //
//////////////////////
void ThinShells::mcVisualization(const string& innerFilename, const V3i& innerResolution,
	const string& outerFilename, const V3i& outerResolution,
	const string& isoFilename, const V3i& isoResolution) const
{
	V3d gridOrigin = modelBoundingBox.boxOrigin;
	V3d gridWidth = modelBoundingBox.boxWidth;

	//if (nodeWidthArray.empty())
	//{
	//	//nodeWidthArray.reserve(svo.numTreeNodes);
	//	auto& svoNodeArray = svo.svoNodeArray;
	//	std::transform(svoNodeArray.begin(), svoNodeArray.end(), std::back_inserter(nodeWidthArray),
	//		[](SVONode node) {
	//			return Eigen::Vector3d(node.width, node.width, node.width);
	//		});
	//}

	if (!outerFilename.empty() && outerShellIsoVal != -DINF)
	{
		cout << "\n[MC] Extract outer shell by MarchingCubes..." << endl;
		MC::marching_cubes(svo.nodeVertexArray, svo.numTreeNodes, nodeWidthArray,
			svo.numNodeVerts, lambda, make_double3(gridOrigin), make_double3(gridWidth),
			make_uint3(outerResolution), outerShellIsoVal, outerFilename);
		cout << "=====================\n";
	}

	if (!innerFilename.empty() && innerShellIsoVal != -DINF)
	{
		cout << "\n[MC] Extract inner shell by MarchingCubes..." << endl;
		MC::marching_cubes(svo.nodeVertexArray, svo.numTreeNodes, nodeWidthArray,
			svo.numNodeVerts, lambda, make_double3(gridOrigin), make_double3(gridWidth),
			make_uint3(innerResolution), innerShellIsoVal, innerFilename);
		cout << "=====================\n";
	}

	if (!isoFilename.empty())
	{
		cout << "\n[MC] Extract isosurface by MarchingCubes..." << endl;
		MC::marching_cubes(svo.nodeVertexArray, svo.numTreeNodes, nodeWidthArray,
			svo.numNodeVerts, lambda, make_double3(gridOrigin), make_double3(gridWidth),
			make_uint3(isoResolution), .0, isoFilename);
		cout << "=====================\n";
	}
}

void ThinShells::textureVisualization(const string& filename) const
{
	writeTexturedObjFile(filename, bSplineVal);
}

//////////////////////
//    Application   //
//////////////////////
// 测试专用
inline void ThinShells::prepareTestDS()
{
	const vector<SVONode>& svoNodeArray = svo.svoNodeArray;
	const vector<node_vertex_type>& allNodeVertexArray = svo.nodeVertexArray;

	depthMorton2Nodes.clear(); depthVert2Idx.clear();

	depthMorton2Nodes.resize(treeDepth);
	depthVert2Idx.resize(treeDepth);

	// 每一层节点莫顿码与其(全局)下标的映射
	size_t _esumNodes = 0;
	const auto& depthNumNodes = svo.depthNumNodes;
	for (int d = 0; d < treeDepth; ++d)
	{
		vector<size_t> d_nodeIdx(depthNumNodes[d]);
		std::iota(d_nodeIdx.begin(), d_nodeIdx.end(), 0);

		std::transform(svoNodeArray.begin() + _esumNodes, svoNodeArray.begin() + _esumNodes + depthNumNodes[d],
			d_nodeIdx.begin(), std::inserter(depthMorton2Nodes[d], depthMorton2Nodes[d].end()),
			[_esumNodes](const SVONode& node, const size_t& idx) {
				return std::make_pair(node.mortonCode, _esumNodes + idx);
			});
		_esumNodes += depthNumNodes[d];
	}

	// 每层顶点到顶点下标(全局)的映射
	const auto& esumDepthNodeVerts = svo.esumDepthNodeVerts;
	for (int d = 0; d < treeDepth; ++d)
	{
		const size_t d_numVerts = svo.depthNodeVertexArray[d].size();
		vector<size_t> d_numVertIdx(d_numVerts);
		std::iota(d_numVertIdx.begin(), d_numVertIdx.end(), 0);

		const size_t& d_esumNodeVerts = esumDepthNodeVerts[d]; // 顶点数量的exclusive scan
		std::transform(allNodeVertexArray.begin() + d_esumNodeVerts, allNodeVertexArray.begin() + d_esumNodeVerts + d_numVerts,
			d_numVertIdx.begin(), std::inserter(depthVert2Idx[d], depthVert2Idx[d].end()),
			[d_esumNodeVerts](const node_vertex_type& val, const size_t& idx) {
				return std::make_pair(val.first, d_esumNodeVerts + idx);
			});
	}
}

vector<int> ThinShells::multiPointQuery(const vector<V3d>& points, double& time, const int& session, const Test::type& choice)
{
	test_type test = (test_type)choice;

	size_t numPoints = points.size();
	vector<int> result;
	if (innerShellIsoVal == -DINF || outerShellIsoVal == -DINF) { printf("Error: You must create shells first!\n"); return result; }
	const vector<SVONode>& svoNodeArray = svo.svoNodeArray;
	VXd q_bSplineVal;

	const vector<node_vertex_type>& allNodeVertexArray = svo.nodeVertexArray;
	const size_t& numNodeVerts = svo.numNodeVerts;
	const V3d& boxOrigin = modelBoundingBox.boxOrigin;
	const V3d& boxEnd = modelBoundingBox.boxEnd;
	const V3d& boxWidth = modelBoundingBox.boxWidth;
	const Eigen::Array3d minRange = boxOrigin - boxWidth;
	const Eigen::Array3d maxRange = boxEnd + boxWidth;

	auto mt_cpuTest = [&]()
	{
#pragma omp parallel
		for (size_t i = 0; i < numPoints; ++i)
		{
			const V3d& point = points[i];
			if ((point.array() < minRange).any() || (point.array() > maxRange).any())
			{
				q_bSplineVal[i] = outerShellIsoVal;
				continue;
			}
			double sum = .0;
#pragma omp parallel for reduction(+ : sum)
			for (int j = 0; j < numNodeVerts; ++j)
			{
				V3d nodeVert = allNodeVertexArray[j].first;
				uint32_t nodeIdx = allNodeVertexArray[j].second;
				double t = BaseFunction4Point(nodeVert, svoNodeArray[nodeIdx].width, point);
				sum += lambda[j] * (BaseFunction4Point(nodeVert, svoNodeArray[nodeIdx].width, point));
			}
			q_bSplineVal[i] = sum;
		}
	};

	auto simd_cpuTest = [&]()
	{
#pragma omp parallel
		for (size_t i = 0; i < numPoints; ++i)
		{
			const V3d& point = points[i];
			if ((point.array() < minRange).any() || (point.array() > maxRange).any())
			{
				q_bSplineVal[i] = outerShellIsoVal;
				continue;
			}
			double sum = .0;
#pragma omp simd simdlen(8)
			for (int j = 0; j < numNodeVerts; ++j)
			{
				V3d nodeVert = allNodeVertexArray[j].first;
				uint32_t nodeIdx = allNodeVertexArray[j].second;
				sum += lambda[j] * (BaseFunction4Point(nodeVert, svoNodeArray[nodeIdx].width, point));
			}
			q_bSplineVal[i] = sum;
		}
	};

	// 通过找范围求b样条值
	auto mt_cpuTest_2 = [&]()
	{
#pragma omp parallel
		for (size_t i = 0; i < numPoints; ++i)
		{
			const V3d& point = points[i];
			if ((point.array() <= boxOrigin.array()).any() || (point.array() >= boxEnd.array()).any())
			{
				q_bSplineVal[i] = outerShellIsoVal;
				continue;
			}

			double sum = 0.0;
			V3i dis = getPointDis(point, boxOrigin, voxelWidth);
			// 在所有格子(包括边缘格子和大格子)的影响范围内
			int maxOffset = 0;
			int searchDepth = 0;
			double searchNodeWidth = voxelWidth;
			for (int i = 0; i < 3; ++i)
			{
				// 在边缘格子影响范围内, 置为0或者svo_gridSize[i] - 1，为了后面莫顿码的计算
				if (dis[i] <= -1) { maxOffset = std::max(maxOffset, std::abs(dis[i])); dis[i] = 0; }
				else if (dis[i] >= svo_gridSize[i]) { maxOffset = std::max(maxOffset, dis[i] - svo_gridSize[i] + 1); dis[i] = svo_gridSize[i] - 1; }
			}
			uint32_t pointMorton = morton::mortonEncode_LUT((uint16_t)dis.x(), (uint16_t)dis.y(), (uint16_t)dis.z());
			maxOffset = nextPow2(maxOffset);
			while (maxOffset >= 2)
			{
				pointMorton /= 8;
				++searchDepth;
				searchNodeWidth *= 2;
				maxOffset >>= 1;
			}

			auto inDmPointsTraits = svo.mq_setInDomainPoints(pointMorton, modelOrigin, searchNodeWidth, searchDepth, depthMorton2Nodes, depthVert2Idx);
			const int nInDmPointsTraits = inDmPointsTraits.size();

#pragma omp parallel for reduction(+ : sum) // for循环中的变量必须得是有符号整型
			for (int j = 0; j < nInDmPointsTraits; ++j)
			{
				const auto& inDmPointTrait = inDmPointsTraits[j];
				sum += lambda[std::get<2>(inDmPointTrait)] * BaseFunction4Point(std::get<0>(inDmPointTrait), std::get<1>(inDmPointTrait), point);
			}
			q_bSplineVal[i] = sum;
		}
	};

	auto simd_cpuTest_2 = [&]()
	{
#pragma omp parallel
		for (size_t i = 0; i < numPoints; ++i)
		{
			const V3d& point = points[i];
			if ((point.array() < minRange).any() || (point.array() > maxRange).any())
			{
				q_bSplineVal[i] = outerShellIsoVal;
				continue;
			}

			double sum = 0.0;
			V3i dis = getPointDis(point, boxOrigin, voxelWidth);
			// 在所有格子(包括边缘格子和大格子)的影响范围内
			int maxOffset = 0;
			int searchDepth = 0;
			double searchNodeWidth = voxelWidth;
			for (int i = 0; i < 3; ++i)
			{
				// 在边缘格子影响范围内, 置为0或者svo_gridSize[i] - 1，为了后面莫顿码的计算
				if (dis[i] <= -1) { maxOffset = std::max(maxOffset, std::abs(dis[i])); dis[i] = 0; }
				else if (dis[i] >= svo_gridSize[i]) { maxOffset = std::max(maxOffset, dis[i] - svo_gridSize[i] + 1); dis[i] = svo_gridSize[i] - 1; }
			}
			uint32_t pointMorton = morton::mortonEncode_LUT((uint16_t)dis.x(), (uint16_t)dis.y(), (uint16_t)dis.z());
			maxOffset = nextPow2(maxOffset);
			while (maxOffset >= 2)
			{
				pointMorton /= 8;
				++searchDepth;
				searchNodeWidth *= 2;
				maxOffset >>= 1;
			}
			auto inDmPoints = svo.mq_setInDomainPoints(pointMorton, modelOrigin, searchNodeWidth, searchDepth, depthMorton2Nodes, depthVert2Idx);
			const int nInDmPointsTraits = inDmPoints.size();

#pragma omp simd simdlen(8)
			for (int j = 0; j < nInDmPointsTraits; ++j)
			{
				const auto& inDmPointTrait = inDmPoints[j];
				sum += lambda[std::get<2>(inDmPointTrait)] * BaseFunction4Point(std::get<0>(inDmPointTrait), std::get<1>(inDmPointTrait), point);
			}
			q_bSplineVal[i] = sum;
		}
	};

	q_bSplineVal.resize(numPoints);
	TimerInterface* timer; createTimer(&timer);
	switch (test)
	{
	case Test::CPU:
		printf("-- [Ours]: Using CPU\n");
		if (depthMorton2Nodes.empty() || depthVert2Idx.empty()) prepareTestDS();
		//mt_cpuTest_2();
		for (int k = 0; k < session; ++k)
		{
			printf("-- [Ours] [Session: %d/%d]", k + 1, session);
			if (k != session - 1) printf("\r");
			else printf("\n");

			startTimer(&timer);

			//mt_cpuTest();
			mt_cpuTest_2();

			stopTimer(&timer);
		}
		time = getAverageTimerValue(&timer) * 1e-3;
		break;
	case Test::CPU_SIMD:
		printf("-- [Ours]: Using CPU-SIMD\n");
		if (depthMorton2Nodes.empty() || depthVert2Idx.empty()) prepareTestDS();
		simd_cpuTest_2();
		for (int k = 0; k < session; ++k)
		{
			printf("-- [Ours] [Session: %d/%d]", k + 1, session);
			if (k != session - 1) printf("\r");
			else printf("\n");

			startTimer(&timer);

			//simd_cpuTest();
			simd_cpuTest_2();

			stopTimer(&timer);
		}
		time = getAverageTimerValue(&timer) * 1e-3;
		break;
	default:
	case Test::CUDA:
		printf("-- [Ours]: Using CUDA\n");
		cuAcc::cpPointQuery(points.size(), svo.numNodeVerts, svo.numTreeNodes, minRange, maxRange,
			points, svo.nodeVertexArray, nodeWidthArray, lambda, outerShellIsoVal, q_bSplineVal);
		for (int k = 0; k < session; ++k)
		{
			printf("-- [Ours] [Session: %d/%d]", k + 1, session);
			if (k != session - 1) printf("\r");
			else printf("\n");

			startTimer(&timer);

			/// TODO: 还没改成点超过bbox就不算b样条值的版本
			cuAcc::cpPointQuery(points.size(), svo.numNodeVerts, svo.numTreeNodes, minRange, maxRange,
				points, svo.nodeVertexArray, nodeWidthArray, lambda, outerShellIsoVal, q_bSplineVal);

			stopTimer(&timer);
		}
		time = getAverageTimerValue(&timer) * 1e-3;
		break;
	}
	deleteTimer(&timer);

	string t_filename = concatFilePath((string)VIS_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"temp_queryValue.txt");
	std::ofstream temp(t_filename);
	temp << std::setiosflags(std::ios::fixed) << std::setprecision(9) << q_bSplineVal << endl;

	std::transform(q_bSplineVal.begin(), q_bSplineVal.end(), std::back_inserter(result),
		[=](const double& val) {
			if (val >= outerShellIsoVal) return 1;
			else if (val <= innerShellIsoVal) return -1;
			else return 0;
		});

	return result;
}

inline void ThinShells::prepareMoveOnSurface(int& ac_treeDepth,
	vector<vector<V3d>>& nodeOrigin,
	vector<std::map<uint32_t, size_t>>& morton2Nodes,
	vector<vector<std::array<double, 8>>>& nodeBSplineVal,
	vector<double>& nodeWidth)
{
	if (treeDepth < 1) { fprintf(stderr, "Error: SVO is empty!"); return; }

	// 初始化数据结构
	const vector<SVONode>& nodeArray = svo.svoNodeArray;
	const vector<size_t>& depthNumNodes = svo.depthNumNodes;

	ac_treeDepth = treeDepth > 2 ? 2 : treeDepth;
	morton2Nodes.resize(ac_treeDepth); // 2层即可
	nodeWidth.resize(ac_treeDepth);
	nodeOrigin.resize(ac_treeDepth);

	vector<size_t> esumDepthNumNodes(ac_treeDepth);
	std::exclusive_scan(depthNumNodes.begin(), depthNumNodes.begin() + ac_treeDepth, esumDepthNumNodes.begin(), 0);
	esumDepthNumNodes.emplace_back((*esumDepthNumNodes.rbegin()) + depthNumNodes[ac_treeDepth - 1]); // 插入所有ac_treeDepth层的节点数

	for (int d = 0; d < ac_treeDepth; ++d)
	{
		vector<size_t> d_numNodesIdx(depthNumNodes[d]);
		std::iota(d_numNodesIdx.begin(), d_numNodesIdx.end(), 0);

		std::transform(nodeArray.begin() + esumDepthNumNodes[d], nodeArray.begin() + esumDepthNumNodes[d] + depthNumNodes[d],
			d_numNodesIdx.begin(), std::inserter(morton2Nodes[d], morton2Nodes[d].end()),
			[](const SVONode& node, const size_t& idx) {
				return std::make_pair(node.mortonCode, idx);
			});

		std::transform(nodeArray.begin() + esumDepthNumNodes[d], nodeArray.begin() + esumDepthNumNodes[d] + depthNumNodes[d],
			std::inserter(nodeOrigin[d], nodeOrigin[d].end()),
			[](const SVONode& node) {
				return node.origin;
			});

		nodeWidth[d] = (*(nodeArray.begin() + esumDepthNumNodes[d])).width;
	}

	const vector<vector<node_vertex_type>>& depthNodeVertexArray = svo.depthNodeVertexArray;
	vector<std::map<V3d, double>> nodeVertex2BSplineVal(ac_treeDepth);

	nodeBSplineVal.resize(ac_treeDepth);

	//#pragma omp parallel
	for (int d = 0; d < ac_treeDepth; ++d)
	{
		const size_t d_numNodeVerts = depthNodeVertexArray[d].size();
		VXd nodeVertBSplineVal(d_numNodeVerts);

		vector<V3d> nodeVertex;
		// 获得所有节点的顶点数组
		std::transform(depthNodeVertexArray[d].begin(), depthNodeVertexArray[d].end(),
			std::back_inserter(nodeVertex), [](const node_vertex_type& val) { return val.first; });
		// 计算这些顶点的b样条值
		cuAcc::cpBSplineVal(d_numNodeVerts, svo.numNodeVerts, svo.numTreeNodes, nodeVertex,
			svo.nodeVertexArray, nodeWidthArray, lambda, nodeVertBSplineVal);
		// 将顶点与b样条值一一对应起来
		std::transform(nodeVertex.begin(), nodeVertex.end(), nodeVertBSplineVal.begin(),
			std::inserter(nodeVertex2BSplineVal[d], nodeVertex2BSplineVal[d].end()),
			[](const V3d& vert, const double& val) {
				return std::make_pair(vert, val);
			});

		// 第d层的节点数
		const size_t d_numNodes = depthNumNodes[d];
		//#pragma omp critical
		{
			nodeBSplineVal[d].resize(d_numNodes);
		}
		//#pragma omp for nowait
		for (int j = 0; j < d_numNodes; ++j)
		{
			// 第d层的第j个节点
			const SVONode& node = *(nodeArray.begin() + esumDepthNumNodes[d] + j);
			// 获取第d层的第j个节点的八个顶点
			for (int k = 0; k < 8; ++k)
			{
				const int xOffset = k & 1;
				const int yOffset = (k >> 1) & 1;
				const int zOffset = (k >> 2) & 1;

				// 第d层的第j个节点的第k个顶点坐标
				V3d corner = (node.origin) + (node.width) * V3d(xOffset, yOffset, zOffset);
				//#pragma omp critical
				{
					// 得到第d层的第j个节点的第k个顶点与其b样条值的对应
					nodeBSplineVal[d][j][k] = nodeVertex2BSplineVal[d].at(corner);
				}
			}
		}
	}
}

void ThinShells::moveOnSurface(const V3d& modelVert, const V3d& v, const size_t& max_move_cnt)
{
	// 准备数据结构
	int ac_treeDepth;
	vector<vector<V3d>> nodeOrigin;
	vector<double> nodeWidth;
	vector<std::map<uint32_t, size_t>> morton2Nodes;
	vector<vector<std::array<double, 8>>> nodeBSplineVal;
	prepareMoveOnSurface(ac_treeDepth, nodeOrigin, morton2Nodes, nodeBSplineVal, nodeWidth);

	// 找到点所属格子的莫顿码
	auto getPointNode = [=](const V3d& vert, const int& depth)->uint32_t
	{
		V3i dis = getPointDis(vert, modelOrigin, nodeWidth[depth]);
		return morton::mortonEncode_LUT((uint16_t)dis.x(), (uint16_t)dis.y(), (uint16_t)dis.z());
	};
	// 找到点所属格子的原点、宽度以及八个顶点的b样条值
	auto getNodeBSplineVal = [=](const V3d& vert, bool& flag)->std::tuple<V3d, double, std::array<double, 8>>
	{
		// 如果不在第0层那肯定就在第1层
		for (int d = 0; d < ac_treeDepth; ++d)
		{
			uint32_t nodeMorton = getPointNode(vert, d);
			if (morton2Nodes[d].find(nodeMorton) != morton2Nodes[d].end())
			{
				size_t nodeIdx = morton2Nodes[d].at(nodeMorton); // at() is thread safe
				//std::cout << "width = " << nodeWidth[d] << std::endl;
				flag = true;
				return std::make_tuple(nodeOrigin[d].at(nodeIdx), nodeWidth[d], nodeBSplineVal[d].at(nodeIdx));
			}
		}
	};
	// 得到点的b样条值
	auto getPointBSplineVal = [=](const V3d& vert)->double
	{
		bool flag = false;
		auto p = getNodeBSplineVal(vert, flag);
		if (!flag) return DINF;
		V3d offset = (vert - std::get<0>(p)) / std::get<1>(p);
		//std::cout << "std::get<1>(p) = " << std::get<1>(p) << ", offset = " << offset.transpose() << std::endl;
		return tri_lerp<double>(std::get<2>(p), offset.x(), offset.y(), offset.z());
	};

	// 可视化输出
	const string saveDir = "PointMove";
	auto writePoint = [=](const V3d& point, const size_t& cnt)
	{
		string filename = concatFilePath((string)VIS_DIR, modelName, uniformDir, std::to_string(treeDepth),
			saveDir, std::to_string(cnt) + (string)".xyz");
		checkDir(filename);
		std::ofstream out(filename);
		if (!out) { fprintf(stderr, "[I/O] Error: File %s could not be opened!", filename.c_str()); return; }
		gvis::write_vertex_to_xyz(out, point);
	};
	// 输出初始点
	size_t valid_move = 0;
	printf("Init point at (%lf, %lf, %lf)\n", modelVert.x(), modelVert.y(), modelVert.z());
	writePoint(modelVert, valid_move);

	// 开始沿v移动
	V3d point = modelVert;
	const V3d speed = v;
	V3d dir = v.normalized();
	double point_bSplineVal = .0;
	V3d toOuterShell = V3d(), toInnerShell = V3d();
	V3d zeroVec = V3d();

	std::default_random_engine e(time(0)); // current time as seed
	std::normal_distribution<double> n(-1, 1);
	while (valid_move < max_move_cnt)
	{
		V3d new_point = point;
		bool flag = false;
		do
		{
			new_point = point + ((speed.array() * dir.array()).matrix());
			point_bSplineVal = getPointBSplineVal(new_point);
			//std::cout << "new_point = " << new_point.transpose() << ", val = " << point_bSplineVal << std::endl;
			//system("pause");

			// 随机一个新的方向
			e.seed(time(0));
			dir = Eigen::Matrix<double, 3, 1>::Zero(3, 1)
				.unaryExpr([&](double val) { return static_cast<double>(round((n(e) + 1e-6) * 1e6) / 1e6); });
		} while (point_bSplineVal > 0.01 || point_bSplineVal < -0.01);
		++valid_move;

		//toOuterShell = ((outerShellIsoVal - point_bSplineVal) * dir).array() / speed.array();
		//toInnerShell = ((innerShellIsoVal - point_bSplineVal) * dir).array() / speed.array();;
		//if (toOuterShell.isApprox(zeroVec, 1e-9) || toInnerShell.isApprox(zeroVec, 1e-9)) // 本身就在外壳或者内壳上
		//{
		//	// 随机一个新的方向
		//	dir = Eigen::Matrix<double, 3, 1>::Zero(3, 1)
		//		.unaryExpr([&](double val) { return static_cast<double>(round((n(e) + 1e-6) * 1e6) / 1e6); });
		//}
		//
		//V3d dis;
		//double new_vert_bSplineVal;
		//do
		//{
		//	// 随机一个新的方向
		//	dir = Eigen::Matrix<double, 3, 1>::Zero(3, 1)
		//		.unaryExpr([&](double val) { return static_cast<double>(round((n(e) + 1e-6) * 1e6) / 1e6); });
		//	dis = (speed.array() * dir.array()).matrix();
		//
		//	if (fabs(dir.x()) > 1e-6) new_vert_bSplineVal = dis.x() / dir.x() + point_bSplineVal;
		//	else if (fabs(dir.y()) > 1e-6) new_vert_bSplineVal = dis.y() / dir.y() + point_bSplineVal;
		//	else if (fabs(dir.z()) > 1e-6) new_vert_bSplineVal = dis.z() / dir.z() + point_bSplineVal;
		//
		//	std::cout << new_vert_bSplineVal << std::endl;
		//} while (new_vert_bSplineVal > outerShellIsoVal || new_vert_bSplineVal < innerShellIsoVal);
		//
		//++valid_move;
		//point = point + dis;
		//point_bSplineVal = new_vert_bSplineVal;

		point = new_point;
		printf("#%llu: point at (%lf, %lf, %lf), bSplineVal = %lf\n", valid_move, point.x(), point.y(), point.z(), point_bSplineVal);
		/*if (valid_move != max_move_cnt - 1) printf("\r");
		else printf("\n");*/
		writePoint(point, valid_move);
	}
}

void ThinShells::launchParticleSystem(const int& maxIterations, const std::string& out_file)
{
	numParticles = nModelVerts;
	particleArray = m_V;
	//#pragma omp parallel for reduction(+ : embedOmiga) // 只有变量或静态数据成员才能用于 data - sharing 子句中
	for (int i = 0; i < nModelTris; ++i)
		embedOmiga += modelTris[i].area;
	theta = c_theta * std::sqrt(embedOmiga / numParticles);

	lbfgs_optimization(maxIterations, out_file);
}

#include <LBFGS.h>
#include <igl/knn.h>
#include "KNNHelper.h"

MXd ThinShells::getSurfacePointNormal(const MXd& queryPointMat)
{
	const int numPoints = queryPointMat.rows();
	MXd pointNormal(numPoints, 3);

	//if (depthMorton2Nodes.empty() || depthVert2Idx.empty()) prepareTestDS();

#pragma omp parallel
	for (int i = 0; i < numPoints; ++i)
	{
		const V3d& point = queryPointMat.row(i);

		V3d gradient;
		V3i dis = getPointDis(point, modelOrigin, voxelWidth);
		double searchNodeWidth = voxelWidth;
		uint32_t pointMorton = morton::mortonEncode_LUT((uint16_t)dis.x(), (uint16_t)dis.y(), (uint16_t)dis.z());

		auto inDmPointsTraits = svo.mq_setInDomainPoints(pointMorton, modelOrigin, searchNodeWidth, 0, depthMorton2Nodes, depthVert2Idx);
		const int nInDmPointsTraits = inDmPointsTraits.size();

		//#pragma omp parallel for reduction(+ : gradient) // for循环中的变量必须得是有符号整型
		for (int j = 0; j < nInDmPointsTraits; ++j)
		{
			const auto& inDmPointTrait = inDmPointsTraits[j];
			gradient += lambda[std::get<2>(inDmPointTrait)] * de_BaseFunction4Point(std::get<0>(inDmPointTrait), std::get<1>(inDmPointTrait), point);
		}
		pointNormal.row(i) = (gradient.normalized());
	}

	return pointNormal;
}

MXd ThinShells::getPointNormal(const MXd& queryPointMat)
{
	const int numPoints = queryPointMat.rows();
	MXd pointNormal(numPoints, 3);
	pointNormal.setZero();
	VXd q_bSplineVal(numPoints);
	q_bSplineVal.setZero();

	const V3d& boxOrigin = modelBoundingBox.boxOrigin;
	const V3d& boxEnd = modelBoundingBox.boxEnd;
	const V3d& boxWidth = modelBoundingBox.boxWidth;
	/*const Eigen::Array3d minRange = boxOrigin - boxWidth;
	const Eigen::Array3d maxRange = boxEnd + boxWidth;*/

#pragma omp parallel
	for (size_t i = 0; i < numPoints; ++i)
	{
		const V3d& point = queryPointMat.row(i);
		if ((point.array() < boxOrigin.array()).any() || (point.array() > boxEnd.array()).any())
		{
			std::cout << "Point is outside boundingbox!\n";
			//std::cout << "point = " << point.transpose() << std::endl;
			continue; // 要保证在boundingbox里面
		}
		//if ((point.array() <= minRange).any() || (point.array() >= maxRange).any()) continue;

		V3d gradient; gradient.setZero();
		V3i dis = getPointDis(point, boxOrigin, voxelWidth);
		// 在所有格子(包括边缘格子和大格子)的影响范围内
		//int maxOffset = 0;
		int searchDepth = 0;
		double searchNodeWidth = voxelWidth;
		uint32_t pointMorton = morton::mortonEncode_LUT((uint16_t)dis.x(), (uint16_t)dis.y(), (uint16_t)dis.z());

		auto inDmPointsTraits = svo.mq_setInDomainPoints(pointMorton, modelOrigin, searchNodeWidth, searchDepth, depthMorton2Nodes, depthVert2Idx);
		const int nInDmPointsTraits = inDmPointsTraits.size();

		//#pragma omp parallel for reduction(+ : gradient) // for循环中的变量必须得是有符号整型，reduction子句中的变量必须是标量算术类型
		for (int j = 0; j < nInDmPointsTraits; ++j)
		{
			const auto& inDmPointTrait = inDmPointsTraits[j];
			gradient += lambda[std::get<2>(inDmPointTrait)] * de_BaseFunction4Point(std::get<0>(inDmPointTrait), std::get<1>(inDmPointTrait), point);
		}
		pointNormal.row(i) = (gradient.normalized());
	}

	return pointNormal;
}

VXd ThinShells::getPointBSplineVal(const MXd& queryPointMat)
{
	const int numPoints = queryPointMat.rows();
	VXd q_bSplineVal;
	q_bSplineVal.setZero();

	if (depthMorton2Nodes.empty() || depthVert2Idx.empty()) prepareTestDS();

	const V3d& boxOrigin = modelBoundingBox.boxOrigin;
	const V3d& boxEnd = modelBoundingBox.boxEnd;
	const V3d& boxWidth = modelBoundingBox.boxWidth;
	const Eigen::Array3d minRange = boxOrigin - boxWidth;
	const Eigen::Array3d maxRange = boxEnd + boxWidth;

#pragma omp parallel
	for (size_t i = 0; i < numPoints; ++i)
	{
		const V3d& point = queryPointMat.row(i);
		if ((point.array() <= boxOrigin.array()).any() || (point.array() >= boxEnd.array()).any())
		{
			std::cout << "[BSplineVal] Point is outside boundingbox!\n";
			continue; // 要保证在boundingbox里面
		}

		double sum = 0.0;
		V3i dis = getPointDis(point, boxOrigin, voxelWidth);
		// 在所有格子(包括边缘格子和大格子)的影响范围内
		int maxOffset = 0;
		int searchDepth = 0;
		double searchNodeWidth = voxelWidth;
		//for (int i = 0; i < 3; ++i)
		//{
		//	// 在边缘格子影响范围内, 置为0或者svo_gridSize[i] - 1，为了后面莫顿码的计算
		//	if (dis[i] <= -1) { maxOffset = std::max(maxOffset, std::abs(dis[i])); dis[i] = 0; }
		//	else if (dis[i] >= svo_gridSize[i]) { maxOffset = std::max(maxOffset, dis[i] - svo_gridSize[i] + 1); dis[i] = svo_gridSize[i] - 1; }
		//}
		uint32_t pointMorton = morton::mortonEncode_LUT((uint16_t)dis.x(), (uint16_t)dis.y(), (uint16_t)dis.z());
		/*maxOffset = nextPow2(maxOffset);
		while (maxOffset >= 2)
		{
			pointMorton /= 8;
			++searchDepth;
			searchNodeWidth *= 2;
			maxOffset >>= 1;
		}*/

		auto inDmPointsTraits = svo.mq_setInDomainPoints(pointMorton, modelOrigin, searchNodeWidth, searchDepth, depthMorton2Nodes, depthVert2Idx);
		const int nInDmPointsTraits = inDmPointsTraits.size();

#pragma omp parallel for reduction(+ : sum) // for循环中的变量必须得是有符号整型
		for (int j = 0; j < nInDmPointsTraits; ++j)
		{
			const auto& inDmPointTrait = inDmPointsTraits[j];
			sum += lambda[std::get<2>(inDmPointTrait)] * BaseFunction4Point(std::get<0>(inDmPointTrait), std::get<1>(inDmPointTrait), point);
		}
		q_bSplineVal[i] = sum;
	}

	return q_bSplineVal;
}

inline std::pair<VXd, MXd> ThinShells::getPointValGradient(const MXd& before_queryPointMat, const MXd& queryPointMat)
{
	const int numPoints = queryPointMat.rows();
	MXd pointGradient(numPoints, 3);
	pointGradient.setZero();
	VXd q_bSplineVal(numPoints);
	q_bSplineVal.setZero();

	const V3d& boxOrigin = modelBoundingBox.boxOrigin;
	const V3d& boxEnd = modelBoundingBox.boxEnd;
	const V3d& boxWidth = modelBoundingBox.boxWidth;
	//std::cout << "boxOrigin = " << boxOrigin.transpose() << ", boxEnd = " << boxEnd << std::endl;
	const Eigen::Array3d minRange = boxOrigin - boxWidth;
	const Eigen::Array3d maxRange = boxEnd + boxWidth;

#pragma omp parallel
	for (size_t i = 0; i < numPoints; ++i)
	{
		V3d point = queryPointMat.row(i);
		V3d before_point = before_queryPointMat.row(i);
		while ((point.array() < boxOrigin.array()).any() || (point.array() > boxEnd.array()).any())
		{
			point = (point + before_point) / 2.0;
			//std::cout << "Point is outside boundingbox!\n";
			//std::cout << "point = " << point.transpose() << std::endl;
			//continue; // 要保证在boundingbox里面
		}
		//if ((point.array() <= minRange).any() || (point.array() >= maxRange).any()) continue;

		V3d gradient; gradient.setZero();
		double sum = 0.0;
		V3i dis = getPointDis(point, boxOrigin, voxelWidth);
		// 在所有格子(包括边缘格子和大格子)的影响范围内
		//int maxOffset = 0;
		int searchDepth = 0;
		double searchNodeWidth = voxelWidth;
		//for (int i = 0; i < 3; ++i)
		//{
		//	// 在边缘格子影响范围内, 置为0或者svo_gridSize[i] - 1，为了后面莫顿码的计算
		//	if (dis[i] <= -1) { maxOffset = std::max(maxOffset, std::abs(dis[i])); dis[i] = 0; }
		//	else if (dis[i] >= svo_gridSize[i]) { maxOffset = std::max(maxOffset, dis[i] - svo_gridSize[i] + 1); dis[i] = svo_gridSize[i] - 1; }
		//}
		uint32_t pointMorton = morton::mortonEncode_LUT((uint16_t)dis.x(), (uint16_t)dis.y(), (uint16_t)dis.z());
		/*maxOffset = nextPow2(maxOffset);
		while (maxOffset >= 2)
		{
			pointMorton /= 8;
			++searchDepth;
			searchNodeWidth *= 2;
			maxOffset >>= 1;
		}*/

		auto inDmPointsTraits = svo.mq_setInDomainPoints(pointMorton, modelOrigin, searchNodeWidth, searchDepth, depthMorton2Nodes, depthVert2Idx);
		const int nInDmPointsTraits = inDmPointsTraits.size();

		//#pragma omp parallel for reduction(+ : gradient) // for循环中的变量必须得是有符号整型，reduction子句中的变量必须是标量算术类型
		for (int j = 0; j < nInDmPointsTraits; ++j)
		{
			const auto& inDmPointTrait = inDmPointsTraits[j];
			sum += lambda[std::get<2>(inDmPointTrait)] * BaseFunction4Point(std::get<0>(inDmPointTrait), std::get<1>(inDmPointTrait), point);
			gradient += lambda[std::get<2>(inDmPointTrait)] * de_BaseFunction4Point(std::get<0>(inDmPointTrait), std::get<1>(inDmPointTrait), point);
		}
		q_bSplineVal[i] = sum;
		pointGradient.row(i) = gradient;
	}

	return std::make_pair(q_bSplineVal, pointGradient);
}

inline MXd ThinShells::getProjectPoint(const MXd& before_queryPointMat, const MXd& queryPointMat, const int& iter)
{
	//const auto& nodeArray = svo.svoNodeArray;
	//SVONode relativeNode = *nodeArray.crbegin();
	//V3d relativeNodeOrigin;
	//double relativeNodeHalfWidth;
	//auto getOffsetInNode = [=](const V3d& point)->uint32_t
	//{
	//	V3i offset = ((point - relativeNodeOrigin).array() / relativeNodeHalfWidth).cast<int>();
	//	return morton::mortonEncode_LUT((uint16_t)offset.x(), (uint16_t)offset.y(), (uint16_t)offset.z()); // 0~7
	//};
	//// 找到所处的最细的叶子节点
	//while (relativeNode.mortonCode != _UI32_MAX && !(relativeNode.isLeaf))
	//{
	//	relativeNodeOrigin = relativeNode.origin;
	//	relativeNodeHalfWidth = relativeNode.width / 2;
	//
	//	const uint32_t childIdx = getOffsetInNode(point);
	//	relativeNode = nodeArray[relativeNode.childs[childIdx]];
	//}
	const int numPoints = queryPointMat.rows();
	MXd proj_point(numPoints, 3);
	proj_point = queryPointMat;

	int n = 0;
	while (n < iter)
	{
		const auto pointValGradient = getPointValGradient(before_queryPointMat, proj_point);
		VXd pointBSplineVal = pointValGradient.first;
		MXd pointGradient = pointValGradient.second;

		//#pragma omp parallel for
		for (int i = 0; i < numPoints; ++i)
		{
			if (pointBSplineVal(i) <= outerShellIsoVal && pointBSplineVal(i) >= innerShellIsoVal) continue;

			const V3d& point = proj_point.row(i);
			V3d gradient = pointGradient.row(i);

			if (gradient.isApprox(V3d(0, 0, 0))) std::cout << "gradient is 0!\n";

			double k = -pointBSplineVal(i) / (gradient.squaredNorm());

			proj_point.row(i) = k * gradient + point;
		}
		++n;
	}

	return proj_point;
}

void ThinShells::lbfgs_optimization(const int& maxIterations, const std::string& out_file)
{
	using namespace LBFGSpp;

	if (depthMorton2Nodes.empty() || depthVert2Idx.empty()) prepareTestDS();

	printf("[LBFGS] Optimizing particle system...\n");

	// Solver's param
	LBFGSParam<double> param;
	param.epsilon = 1e-5;
	param.max_iterations = 1;
	//param.max_linesearch = 5000;

	// Project particles to the surface
	MXd proj_particleMat;

	// Compute normal
	MXd normal;

	// Nearest points
	std::vector<MXd> kd_neighPointList;

	std::vector<std::vector<V3d>> neighPointList(numParticles);
	std::vector<std::vector<V3d>> fineNodeParticle(svo.numTreeNodes);
	std::map<V3d, uint32_t> particle2Node;

	auto updateNeighbor = [&](const MXd& queryPointMat)
	{
		for (auto& innerVec : neighPointList) innerVec.clear();
		neighPointList.clear(); neighPointList.resize(numParticles);

		for (auto& innerVec : fineNodeParticle) innerVec.clear();
		fineNodeParticle.clear(); fineNodeParticle.resize(svo.numTreeNodes);

		particle2Node.clear();

		int t = 0;
		//#pragma omp parallel for
		for (int i = 0; i < numParticles; ++i)
		{
			const V3d& particle = queryPointMat.row(i);
			const V3i dis = getPointDis(particle, modelOrigin, voxelWidth);
			uint32_t mortonCode = morton::mortonEncode_LUT((uint16_t)dis.x(), (uint16_t)dis.y(), (uint16_t)dis.z());
			int depth = 0;
			while (depth < treeDepth && depthMorton2Nodes[depth].find(mortonCode) == depthMorton2Nodes[depth].end())
			{
				++depth;
				mortonCode /= 8;
			}
			if (depth == treeDepth)
			{
				printf("out of bounding box\n");
				continue;
			}

			uint32_t svoNodeIdx = depthMorton2Nodes[depth].at(mortonCode);

			if (svoNodeIdx >= fineNodeParticle.size()) printf("111\n");

			//#pragma omp critical
			{
				fineNodeParticle[svoNodeIdx].emplace_back(particle);
				particle2Node.emplace(std::make_pair(particle, svoNodeIdx));
				//particle2Node.insert(std::make_pair(particle, svoNodeIdx));
			}
		}

		for (int i = 0; i < numParticles; ++i)
		{
			const V3d& particle = queryPointMat.row(i);
			if (particle2Node.find(particle) != particle2Node.end())
			{
				const uint32_t nodeIdx = particle2Node.at(particle);
				if (!fineNodeParticle[nodeIdx].empty())
				{
					for (int j = 0; j < fineNodeParticle[nodeIdx].size(); ++j)
					{
						const V3d j_neiParticle = fineNodeParticle[nodeIdx][j];
						neighPointList[i].emplace_back(j_neiParticle);
					}
				}
			}
		}

	};

	int numSearch = 6;

	// Create function object
	auto optimize_fun = [&](const Eigen::VectorXd& before_particle, Eigen::VectorXd& grad)
	{
		double systemEnergy = .0;

		MXd before_particleMat(numParticles, 3);
		for (size_t i = 0; i < numParticles; ++i)
			before_particleMat.row(i) = Eigen::Vector3d(before_particle(i * 3), before_particle(i * 3 + 1), before_particle(i * 3 + 2));

		// Nearest point search
		/*KDTree kdTree(3, before_particleMat, 100);
		std::vector<MXd> neighPointList;
		knn_helper::getNeighborPoint(kdTree, before_particleMat, numSearch, neighPointList);*/

		//updateNeighbor(before_particleMat);
		//normal = getPointNormal(before_particleMat);

		/* ours */
		//for (int i = 0; i < numParticles; ++i)
		//{
		//	const V3d i_particle = before_particleMat.row(i);
		//	//const V3d i_particle = V3d(before_particle(i * 3), before_particle(i * 3 + 1), before_particle(i * 3 + 2));
		//
		//	// Compute gradient of i'th particle
		//	Eigen::Vector3d i_force; i_force.setZero();
		//
		//	/*for (int j = 0; j < numSearch; ++j)
		//	{
		//		const V3d j_neiParticle = neighPointList[i].row(j);
		//
		//		const double& ij_energy = std::exp(-((i_particle - j_neiParticle).squaredNorm()) / (4 * theta * theta));
		//		systemEnergy += ij_energy;
		//		i_force += ((i_particle - j_neiParticle) / (2 * theta * theta)) * ij_energy;
		//	}*/
		//
		//	for (int j = 0; j < neighPointList[i].size(); ++j)
		//	{
		//		const V3d j_neiParticle = neighPointList[i][j];
		//		if (i_particle == j_neiParticle) continue;
		//
		//		double ij_energy = std::exp(-((i_particle - j_neiParticle).squaredNorm()) / (4 * theta * theta));
		//      //if (fabs(ij_energy) < 1e-20) ij_energy = 0;
		//		systemEnergy += ij_energy;
		//
		//		i_force += ((i_particle - j_neiParticle) / (2 * theta * theta)) * ij_energy;
		//
		//		for (int i = 0; i < i_force.rows(); ++i)
		//			if (std::isnan(i_force(i))) {
		//				std::cout << i_particle.transpose() << ", " << j_neiParticle.transpose() << ", " << ij_energy << std::endl;
		//				std::cout << "#1 nan!\n"; system("pause");
		//			}
		//	}
		//
		//	/*if (particle2Node.find(i_particle) != particle2Node.end())
		//	{
		//		const uint32_t nodeIdx = particle2Node.at(i_particle);
		//		for (int j = 0; j < fineNodeParticle[nodeIdx].size(); ++j)
		//		{
		//			const V3d j_neiParticle = fineNodeParticle[nodeIdx][j];
		//			if (i_particle == j_neiParticle) continue;
		//			const double& ij_energy = std::exp(-((i_particle - j_neiParticle).squaredNorm()) / (4 * theta * theta));
		//			systemEnergy += ij_energy;
		//			i_force += ((i_particle - j_neiParticle) / (2 * theta * theta)) * ij_energy;
		//		}
		//	}*/
		//
		//	// Project 'i_force' to the surface tangent
		//	const V3d i_normal = normal.row(i);
		//	i_force = i_force - ((i_force.dot(i_normal)) * i_normal);
		//	//i_force.normalize();
		//
		//	// Update gradient
		//	grad(i * 3) = i_force.x();
		//	grad(i * 3 + 1) = i_force.y();
		//	grad(i * 3 + 2) = i_force.z();
		//}

		for (int i = 0; i < numParticles; ++i)
		{
			//const Eigen::Vector3d i_particle = Eigen::Vector3d(before_particle(i * 3), before_particle(i * 3 + 1), before_particle(i * 3 + 2));
			const V3d i_particle = before_particleMat.row(i);

			// Compute gradient of i'th particle
			V3d i_force; i_force.setZero();
			for (int j = 0; j < numSearch; ++j)
			{
				const Eigen::Vector3d j_neiParticle = kd_neighPointList[i].row(j);

				const double& ij_energy = std::exp(-((i_particle - j_neiParticle).squaredNorm()) / (4 * theta * theta));
				systemEnergy += ij_energy;
				i_force += ((i_particle - j_neiParticle) / (2 * theta * theta)) * ij_energy;
			}

			// Project 'i_force' to the surface tangent
			const Eigen::Vector3d i_normal = normal.row(i);
			i_force = i_force - (i_force.dot(i_normal)) * i_normal;
			//i_force.normalize();

			// Update gradient
			grad(i * 3) = i_force.x();
			grad(i * 3 + 1) = i_force.y();
			grad(i * 3 + 2) = i_force.z();
		}

		/*std::cout << "Line Search: systemEnergy = " << systemEnergy << std::endl;
		std::cout << "=========\n";*/
		return systemEnergy;
	};

	// Create solver
	LBFGSSolver<double, LineSearchBracketing> solver(param);

	double energy;

	Eigen::VectorXd particle_x;
	int proj_iter = 2;
	for (int iter = 1; iter <= maxIterations; ++iter)
	{
		if (iter == 1) { proj_particleMat = particleArray; normal = m_VN; }
		else normal = getSurfacePointNormal(proj_particleMat);

		if (iter % 2 == 0) proj_iter += 2;

		//updateNeighbor(proj_particleMat);

		// Nearest point search
		neighPointList.clear(); neighPointList.resize(0);
		KDTree kdTree(3, proj_particleMat, 10);
		knn_helper::getNeighborPoint(kdTree, proj_particleMat, numSearch, kd_neighPointList);

		MXd before_proj_particleMat = proj_particleMat;

		Eigen::MatrixXd trans_proj_particleMat = proj_particleMat;
		trans_proj_particleMat.transposeInPlace();
		particle_x = (Eigen::Map<Eigen::VectorXd>(trans_proj_particleMat.data(), numParticles * 3));

		solver.minimize(optimize_fun, particle_x, energy);
		printf("-- [Iter: %d/%d] System energy = %lf\n", iter, maxIterations, energy);

		Eigen::MatrixXd particleMat(numParticles, 3);
		//#pragma omp parallel for
		for (int i = 0; i < numParticles; ++i)
			particleMat.row(i) = Eigen::Vector3d(particle_x(i * 3), particle_x(i * 3 + 1), particle_x(i * 3 + 2));

		const string tempResFile = concatFilePath((string)VIS_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"temp_particle.xyz");
		std::ofstream out1(tempResFile);
		gvis::writePointCloud_xyz(particleMat, out1);
		//system("pause");

		proj_particleMat = getProjectPoint(before_proj_particleMat, particleMat, proj_iter);
		const string tempProjFile = concatFilePath((string)VIS_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"proj_particle.xyz");
		std::ofstream out2(tempProjFile);
		gvis::writePointCloud_xyz(proj_particleMat, out2);
		//system("pause");
	}
	//proj_particleMat = getProjectPoint(proj_particleMat, 10); // 最后的投影
	Eigen::VectorXd sqrD;
	Eigen::VectorXi I;
	Eigen::MatrixXd C;

	// the output sqrD contains the (unsigned) squared distance from each point in P 
	// to its closest point given in C which lies on the element in F given by I
	aabbTree.squared_distance(m_V, m_F, proj_particleMat, sqrD, I, C);
	proj_particleMat = C;

	if (!out_file.empty())
	{
		checkDir(out_file);
		std::ofstream out(out_file);
		if (!out) { fprintf(stderr, "[I/O] Error: File %s could not be opened!", out_file.c_str()); return; }
		printf("[I/O] Output final partile to file: %s\n", out_file.c_str());

		if (getFileExtension(out_file) == ".obj") gvis::writePointCloud(proj_particleMat, out);
		else if (getFileExtension(out_file) == ".xyz") gvis::writePointCloud_xyz(proj_particleMat, out);
		else fprintf(stderr, "[I/O] Error: File %s could not supported!", out_file.c_str());

		out.close();
	}
}