#include "ThinShells.h"
#include "MortonLUT.h"
#include "BSpline.hpp"
#include "utils\IO.hpp"
#include "utils\Timer.hpp"
#include "utils\Common.hpp"
#include "utils\String.hpp"
#include "cuAcc\CUDACompute.h"
#include "utils\cuda\CUDAMath.hpp"
#include "cuAcc\MarchingCubes\MarchingCubes.h"
#include <omp.h>
#include <queue>
#include <iomanip>
#include <numeric>
#include <Eigen\Sparse>
#include <igl\signed_distance.h>

//////////////////////
//  Create  Shells  //
//////////////////////
inline int parallelAxis(const V3d& p1, const V3d& p2)
{
	if (fabs(p1.y() - p2.y()) < 1e-9 && fabs(p1.z() - p2.z()) < 1e-9) return 1; // 与x轴平行
	else if (fabs(p1.x() - p2.x()) < 1e-9 && fabs(p1.z() - p2.z()) < 1e-9) return 2; // 与y轴平行
	else return 3; // 与z轴平行
};

inline V3i ThinShells::getPointDis(const V3d& vert, const V3d& origin, const double& width)
{
	return ((vert - origin).array() / width).cast<int>();
};

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

	std::map<uint32_t, SVONode> morton2FineNode;
	std::transform(nodeArray.begin(), nodeArray.begin() + numFineNodes,
		std::inserter(morton2FineNode, morton2FineNode.end()),
		[](const SVONode& node) {
			return std::make_pair(node.mortonCode, node);
		});

#pragma omp parallel
	for (int i = 0; i < nModelEdges; i++)
	{
		std::vector<V3d> edge_vec_private;

		Eigen::Vector2i e = modelEdges[i];
		V3d p1 = m_V.row(e.x()); V3d p2 = m_V.row(e.y());
		//if (!isLess(p1, p2, std::less<V3d>())) std::swap(p1, p2);

		V3d modelEdgeDir = p2 - p1;

		V3i dis1 = getPointDis(p1, modelOrigin, voxelWidth);
		V3i dis2 = getPointDis(p2, modelOrigin, voxelWidth);

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
					if (morton2FineNode.find(nodeMorton) == morton2FineNode.end()) continue;
					V3d lbbCorner = morton2FineNode.at(nodeMorton).origin; // at() is thread safe
					double width = morton2FineNode.at(nodeMorton).width;

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
					}
					if (isInRange(.0, 1.0, left_t) &&
						isInRange(lbbCorner.x(), lbbCorner.x() + width, (p1 + left_t * modelEdgeDir).x()) &&
						isInRange(lbbCorner.z(), lbbCorner.z() + width, (p1 + left_t * modelEdgeDir).z()))
					{
						//edgeInterPoints.emplace_back(p1 + left_t * modelEdgeDir);
						edge_vec_private.emplace_back(p1 + left_t * modelEdgeDir);
					}
					if (isInRange(.0, 1.0, bottom_t) &&
						isInRange(lbbCorner.x(), lbbCorner.x() + width, (p1 + bottom_t * modelEdgeDir).x()) &&
						isInRange(lbbCorner.y(), lbbCorner.y() + width, (p1 + bottom_t * modelEdgeDir).y()))
					{
						//edgeInterPoints.emplace_back(p1 + bottom_t * modelEdgeDir);
						edge_vec_private.emplace_back(p1 + bottom_t * modelEdgeDir);
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
	cout << "-- 三角形边与node的交点数量：" << edgeInterPoints.size() << endl;

	allInterPoints.insert(allInterPoints.end(), edgeInterPoints.begin(), edgeInterPoints.end());

	// 三角形面与node边线交
	std::cout << "2. Computing the intersections between mesh FACES and node EDGES..." << endl;
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
	cout << "-- 三角形面与node边的交点数量：" << faceInterPoints.size() << endl;

	allInterPoints.insert(allInterPoints.end(), faceInterPoints.begin(), faceInterPoints.end());
	cout << "-- 总交点数量：" << allInterPoints.size() << endl;
}

inline void ThinShells::cpSDFOfTreeNodes()
{
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
}

inline void ThinShells::cpCoefficients()
{
	using SpMat = Eigen::SparseMatrix<double>;
	using Trip = Eigen::Triplet<double>;

	const vector<vector<node_vertex_type>>& depthNodeVertexArray = svo.depthNodeVertexArray;
	const vector<size_t>& esumDepthNodeVerts = svo.esumDepthNodeVerts;
	vector<std::map<V3d, size_t>> nodeVertex2Idx(treeDepth);

	for (int d = 0; d < treeDepth; ++d)
	{
		const size_t d_numNodeVerts = depthNodeVertexArray[d].size();
		vector<size_t> d_nodeVertexIdx(d_numNodeVerts);
		std::iota(d_nodeVertexIdx.begin(), d_nodeVertexIdx.end(), 0);

		std::transform(depthNodeVertexArray[d].begin(), depthNodeVertexArray[d].end(), d_nodeVertexIdx.begin(),
			std::inserter(nodeVertex2Idx[d], nodeVertex2Idx[d].end()),
			[](const node_vertex_type& val, size_t i) {
				return std::make_pair(val.first, i);
			});
	}

	// initial matrix
	const size_t& numNodeVerts = svo.numNodeVerts;
	vector<Trip> matApVal;
#pragma omp parallel
	for (int d = 0; d < treeDepth; ++d)
	{
		const size_t d_numNodeVerts = depthNodeVertexArray[d].size(); // 每层节点的顶点数量
		const size_t& d_esumNodeVerts = esumDepthNodeVerts[d]; // 顶点数量的exclusive scan
#pragma omp for nowait
		for (int i = 0; i < d_numNodeVerts; ++i)
		{
			V3d i_nodeVertex = depthNodeVertexArray[d][i].first;
			uint32_t i_fromNodeIdx = depthNodeVertexArray[d][i].second;

			vector<Trip> private_matApVal;

			//matApVal.emplace_back(Trip(d_esumNodeVerts + i, d_esumNodeVerts + i, 1)); // self
			private_matApVal.emplace_back(Trip(d_esumNodeVerts + i, d_esumNodeVerts + i, 1)); // self

			//#pragma omp parallel for
			for (int j = d - 1; j >= 0; --j)
			{
				if (nodeVertex2Idx[j].find(i_nodeVertex) == nodeVertex2Idx[j].end()) break;
				//matApVal.emplace_back(Trip(d_esumNodeVerts + i, esumDepthNodeVerts[j] + nodeVertex2Idx[j][i_nodeVertex], 1)); // child
				private_matApVal.emplace_back(Trip(d_esumNodeVerts + i, esumDepthNodeVerts[j] + nodeVertex2Idx[j][i_nodeVertex], 1)); // child
			}

			// parent
			auto [inDmPoints, inDmPointsIdx] = svo.setInDomainPoints(i_fromNodeIdx, d + 1, esumDepthNodeVerts, nodeVertex2Idx);
			const int nInDmPoints = inDmPoints.size();

			//#pragma omp parallel for
			for (int k = 0; k < nInDmPoints; ++k)
			{
				double val = BaseFunction4Point(inDmPoints[k].first, inDmPoints[k].second, i_nodeVertex);
				assert(inDmPointsIdx[k] < numNodeVerts, "index of col > numNodeVertex!");
				//if (val != 0) matApVal.emplace_back(Trip(d_esumNodeVerts + i, inDmPointsIdx[k], val));
				if (val != 0)  private_matApVal.emplace_back(Trip(d_esumNodeVerts + i, inDmPointsIdx[k], val));
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

	TimerInterface* timer = nullptr;
	createTimer(&timer);

	Eigen::LeastSquaresConjugateGradient<SpMat> lscg;
	startTimer(&timer);
	lscg.compute(A);
	lambda = lscg.solve(b);

	stopTimer(&timer);
	double time = getElapsedTime(&timer) * 1e-3;
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

namespace {
	vector<V3d> nodeWidthArray;
}
inline void ThinShells::cpBSplineValue()
{
	const uint numAllPoints = nModelVerts + allInterPoints.size();
	std::vector<V3d> pointsData;
	pointsData.insert(pointsData.end(), modelVerts.begin(), modelVerts.end());
	pointsData.insert(pointsData.end(), allInterPoints.begin(), allInterPoints.end());

	if (nodeWidthArray.empty())
	{
		auto& svoNodeArray = svo.svoNodeArray;
		std::transform(svoNodeArray.begin(), svoNodeArray.end(), std::back_inserter(nodeWidthArray),
			[](SVONode node) {
				return Eigen::Vector3d(node.width, node.width, node.width);
			});
	}

	const uint nInterPoints = allInterPoints.size();
	bSplineVal.resize(nModelVerts + nInterPoints);
	bSplineVal.setZero();

	cuAcc::cpBSplineVal(numAllPoints, svo.numNodeVerts, svo.numTreeNodes, pointsData,
		svo.nodeVertexArray, nodeWidthArray, lambda, bSplineVal);

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
	std::cout << "-- innerShellIsoVal: " << innerShellIsoVal << std::endl;
	std::cout << "-- outerShellIsoVal: " << outerShellIsoVal << std::endl;

	//bSplineTree.saveBSplineValue(concatFilePath((string)OUT_DIR, modelName, std::to_string(treeDepth), (string)"bSplineVal.txt"));
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
	saveIntersections("", "");

	cout << "\nComputing discrete SDF of tree nodes..." << endl;
	startTimer(&timer);
	cpSDFOfTreeNodes();
	stopTimer(&timer);
	time = getElapsedTime(&timer) * 1e-3;
	printf("-- Elapsed time: %lf s.\n", time);
	cout << "=====================\n";
	saveSDFValue("");

	cout << "\nComputing coefficients..." << endl;
	startTimer(&timer);
	cpCoefficients();
	stopTimer(&timer);
	time = getElapsedTime(&timer) * 1e-3;
	printf("-- Elapsed time: %lf s.\n", time);
	cout << "=====================\n";
	saveCoefficients("");

	cout << "\nComputing B-Spline value..." << endl;
	startTimer(&timer);
	cpBSplineValue();
	stopTimer(&timer);
	time = getElapsedTime(&timer) * 1e-3;
	printf("-- Elapsed time: %lf s.\n", time);
	cout << "=====================\n";
	saveBSplineValue("");

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
	if (filename.empty()) t_filename = concatFilePath((string)VIS_DIR, modelName, std::to_string(treeDepth), (string)"svo.obj");

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
		t_filename = concatFilePath((string)VIS_DIR, modelName, std::to_string(treeDepth), (string)"edgeInter.xyz");
	cout << "-- Save mesh EDGES and octree Nodes to " << std::quoted(t_filename) << endl;
	saveIntersections(t_filename, edgeInterPoints);

	t_filename = filename_2;
	if (filename_2.empty())
		t_filename = concatFilePath((string)VIS_DIR, modelName, std::to_string(treeDepth), (string)"faceInter.xyz");
	cout << "-- Save mesh FACES and octree node EDGES to " << std::quoted(t_filename) << endl;
	saveIntersections(t_filename, faceInterPoints);
}

void ThinShells::saveSDFValue(const string& filename) const
{
	string t_filename = filename;
	if (filename.empty())
		t_filename = concatFilePath((string)OUT_DIR, modelName, std::to_string(treeDepth), (string)"SDFValue.txt");

	checkDir(t_filename);
	std::ofstream out(t_filename);
	if (!out) { fprintf(stderr, "[I/O] Error: File %s could not open!", filename.c_str()); return; }

	cout << "-- Save SDF value to " << std::quoted(t_filename) << endl;
	for (const auto& val : sdfVal)
		out << val << endl;
	out.close();
}

void ThinShells::saveCoefficients(const string& filename) const
{
	string t_filename = filename;
	if (filename.empty())
		t_filename = concatFilePath((string)OUT_DIR, modelName, std::to_string(treeDepth), (string)"Coefficients.txt");

	checkDir(t_filename);
	std::ofstream out(t_filename);
	if (!out) { fprintf(stderr, "[I/O] Error: File %s could not open!", t_filename.c_str()); return; }

	cout << "-- Save coefficients to " << std::quoted(t_filename) << endl;
	for (const auto& val : lambda)
		out << val << endl;
}

void ThinShells::saveBSplineValue(const string& filename) const
{
	string t_filename = filename;
	if (filename.empty())
		t_filename = concatFilePath((string)OUT_DIR, modelName, std::to_string(treeDepth), (string)"BSplineValue.txt");

	checkDir(t_filename);
	std::ofstream out(t_filename);
	if (!out) { fprintf(stderr, "[I/O] Error: File %s could not open!", t_filename.c_str()); return; }

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

	if (nodeWidthArray.empty())
	{
		//nodeWidthArray.reserve(svo.numTreeNodes);
		auto& svoNodeArray = svo.svoNodeArray;
		std::transform(svoNodeArray.begin(), svoNodeArray.end(), std::back_inserter(nodeWidthArray),
			[](SVONode node) {
				return Eigen::Vector3d(node.width, node.width, node.width);
			});
	}

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


void ThinShells::moveOnSurface(const V3d& modelVert, const V3d& v)
{
	constexpr string saveDir = "PointMove";

	// 输出初始点
	string filename = concatFilePath((string)VIS_DIR, modelName, std::to_string(treeDepth), saveDir, (string)"init.xyz");
	checkDir(filename);
	std::ofstream out(filename);
	if (!out) { fprintf(stderr, "[I/O] Error: File %s could not be opened!", filename.c_str()); return; }
	gvis::write_vertex_to_xyz(out, modelVert);

	// 初始化数据结构
	const vector<SVONode>& nodeArray = svo.svoNodeArray;
	vector<std::map<uint32_t, SVONode>> morton2Nodes(treeDepth);
	for (int i = 0; i < treeDepth; ++i)
	{
		std::transform(nodeArray.begin(), nodeArray.begin() + numFineNodes,
			std::inserter(morton2FineNode, morton2FineNode.end()),
			[](const SVONode& node) {
				return std::make_pair(node.mortonCode, node);
			});
	}
	if (nodeArray.empty() || morton2Nodes.empty()) { fprintf(stderr, "Error: SVO is empty!"); return; }

	const vector<vector<node_vertex_type>>& depthNodeVertexArray = svo.depthNodeVertexArray;
	const vector<size_t>& depthNumNodes = svo.depthNumNodes;
	vector<std::map<V3d, double>> nodeVertex2BSplineVal(treeDepth);

	vector<size_t> esumDepthNumNodes;
	std::exclusive_scan(depthNumNodes.begin(), depthNumNodes.end(), esumDepthNumNodes.end(), 0);
	esumDepthNumNodes.emplace_back(svo.numTreeNodes);

	vector<vector<std::array<double, 8>>> nodeBSplineVal(treeDepth);
#pragma omp parallel for
	for (int d = 0; d < treeDepth; ++d)
	{
		const size_t d_numNodeVerts = depthNodeVertexArray[d].size();
		VXd nodeVertBSplineVal(d_numNodeVerts);

		vector<V3d> nodeVertex;
		std::transform(depthNodeVertexArray[d].begin(), depthNodeVertexArray[d].end(),
			std::back_inserter(nodeVertex), [](const node_vertex_type& val) { return val.first; });

		cuAcc::cpBSplineVal(d_numNodeVerts, svo.numNodeVerts, svo.numTreeNodes, nodeVertex,
			svo.nodeVertexArray, nodeWidthArray, lambda, nodeVertBSplineVal);

		std::transform(nodeVertex.begin(), nodeVertex.end(), nodeVertBSplineVal.begin(),
			std::inserter(nodeVertex2BSplineVal[d], nodeVertex2BSplineVal[d].end()),
			[](const V3d& vert, const double& val) {
				return std::make_pair(vert, val);
			});

#pragma omp critical
		{
			nodeBSplineVal[d].resize(depthNumNodes[d]);
		}
#pragma omp for nowait
		for (int j = 0; j < depthNumNodes[d]; ++j)
		{
			const SVONode& node = *(nodeArray.begin() + esumDepthNumNodes[d] + j);
			for (int k = 0; k < 8; ++k)
			{
				const int xOffset = k & 1;
				const int yOffset = (k >> 1) & 1;
				const int zOffset = (k >> 2) & 1;

				V3d corner = (node.origin) + (node.width) * V3d(xOffset, yOffset, zOffset);
#pragma omp critical
				{
					nodeBSplineVal[d][j][k] = nodeVertex2BSplineVal[d].at(corner); // 得到每个节点八个顶点的sdf值
				}
			}
		}
	}

	// 找到初始点所属的体素格子
	auto getPointNode = [=](const V3d& vert, const int& depth, const V3d& origin, const double& width)->SVONode
	{
		V3i dis = getPointDis(vert, origin, width);
		uint32_t nodeMorton = morton::mortonEncode_LUT((uint16_t)dis.x(), (uint16_t)dis.y(), (uint16_t)dis.z());
		return morton2Nodes[depth].at(nodeMorton); // at() is thread safe
	};
	V3d lbbCorner = getPointNode(modelVert, 0, modelOrigin, voxelWidth).origin;

	// 开始沿v移动
}

void ThinShells::pointProjection(const V3d& point)
{
	const auto& nodeArray = svo.svoNodeArray;

	SVONode relativeNode = *nodeArray.crbegin();
	V3d relativeNodeOrigin;
	double relativeNodeHalfWidth;

	auto getOffsetInNode = [=](const V3d& point)->uint32_t
	{
		V3i offset = ((point - relativeNodeOrigin).array() / relativeNodeHalfWidth).cast<int>();
		return morton::mortonEncode_LUT((uint16_t)offset.x(), (uint16_t)offset.y(), (uint16_t)offset.z()); // 0~7
	};

	// 找到所处的最细的叶子节点
	while (relativeNode.mortonCode != _UI32_MAX && !(relativeNode.isLeaf))
	{
		relativeNodeOrigin = relativeNode.origin;
		relativeNodeHalfWidth = relativeNode.width / 2;

		const uint32_t childIdx = getOffsetInNode(point);
		relativeNode = nodeArray[relativeNode.childs[childIdx]];
	}


}
