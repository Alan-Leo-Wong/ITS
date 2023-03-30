#include "Octree.h"
#include "SDFHelper.h"
#include "BSpline.hpp"
#include "utils\common.hpp"
#include "MarchingCubes.hpp"
#include <queue>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <Windows.h>
#include <algorithm>
#include <Eigen\sparse>
#include <igl\writeOBJ.h>
#include <igl\marching_cubes.h>

//////////////////////
//      Setter      //
//////////////////////
inline void OctreeNode::setCorners()
{
	double minX = boundary.first.x();
	double minY = boundary.first.y();
	double minZ = boundary.first.z();
	double maxX = boundary.second.x();
	double maxY = boundary.second.y();
	double maxZ = boundary.second.z();

	corners[0] = boundary.first;
	corners[1] = V3d(maxX, minY, minZ);
	corners[2] = V3d(minX, maxY, minZ);
	corners[3] = V3d(maxX, maxY, minZ);
	corners[4] = V3d(minX, minY, maxZ);
	corners[5] = V3d(maxX, minY, maxZ);
	corners[6] = V3d(minX, maxY, maxZ);
	corners[7] = boundary.second;
}

inline void OctreeNode::setCornersIdx(map<V3d, vector<PUII>>& corner2IDs)
{
	for (int k = 0; k < 8; ++k)
		corner2IDs[corners[k]].emplace_back(std::make_pair(id, k));
}

inline void OctreeNode::setEdges()
{
	// X
	edges.emplace_back(std::make_pair(corners[0], corners[1]));
	edges.emplace_back(std::make_pair(corners[2], corners[3]));
	edges.emplace_back(std::make_pair(corners[4], corners[5]));
	edges.emplace_back(std::make_pair(corners[6], corners[7]));

	// Y
	edges.emplace_back(std::make_pair(corners[0], corners[2]));
	edges.emplace_back(std::make_pair(corners[1], corners[3]));
	edges.emplace_back(std::make_pair(corners[4], corners[6]));
	edges.emplace_back(std::make_pair(corners[5], corners[7]));

	// Z
	edges.emplace_back(std::make_pair(corners[0], corners[4]));
	edges.emplace_back(std::make_pair(corners[1], corners[5]));
	edges.emplace_back(std::make_pair(corners[2], corners[6]));
	edges.emplace_back(std::make_pair(corners[3], corners[7]));
}

// 判断q_boundary是否在node的影响域内
//inline bool OctreeNode::isInDomain(const PV3d& q_boundary)
//{
//	V3d st_b = q_boundary.first;
//	V3d ed_b = q_boundary.second;
//
//	if (domain[0].x() <= st_b.x() && ed_b.x() <= domain[1].x() &&
//		domain[0].y() <= st_b.y() && ed_b.y() <= domain[1].y() &&
//		domain[0].z() <= st_b.z() && ed_b.z() <= domain[1].z())
//		return true;
//	return false;
//}

//////////////////////
//   Create  Tree   //
//////////////////////
void Octree::createOctree(const BoundingBox& bb, const uint& nPoints, const vector<V3d> modelVerts)
{
	vector<uint> idxOfPoints(nPoints);
	std::iota(idxOfPoints.begin(), idxOfPoints.end(), 0); // 0~pointNum-1

	V3d width = bb.boxWidth;
	PV3d boundary = std::make_pair(bb.boxOrigin, bb.boxEnd);

	root = new OctreeNode(nAllNodes++, 0, width, boundary, idxOfPoints);
	createNode(root, 0, width, boundary, modelVerts, idxOfPoints);
}

inline void Octree::createNode(OctreeNode*& node, const int& depth,
	const V3d& width, const std::pair<V3d, V3d>& boundary,
	const vector<V3d> modelVerts, const vector<uint>& idxOfPoints)
{
	allNodes.emplace_back(node);
	node->setCorners();
	node->setCornersIdx(corner2IDs);

	vector<vector<uint>> childPointsIdx(8);

	V3d b_beg = boundary.first;
	V3d b_end = boundary.second;
	V3d center = (b_beg + b_end) * 0.5;

	if (depth + 1 >= maxDepth || idxOfPoints.empty())
	{
		node->setEdges();
		nLeafNodes++;
		leafNodes.emplace_back(node);
		return;
	}

	node->isLeaf = false;
	node->childs.resize(8, nullptr);
	for (uint idx : idxOfPoints)
	{
		V3d p = modelVerts[idx];

		int whichChild = 0;
		if (p.x() > center.x()) whichChild |= 1;
		if (p.y() > center.y()) whichChild |= 2;
		if (p.z() > center.z()) whichChild |= 4;

		childPointsIdx[whichChild].emplace_back(idx);
	}

	V3d childWidth = width * 0.5;
	for (int i = 0; i < 8; i++)
	{
		const int xOffset = i & 1;
		const int yOffset = (i >> 1) & 1;
		const int zOffset = (i >> 2) & 1;
		V3d childBeg = b_beg + V3d(xOffset * childWidth.x(), yOffset * childWidth.y(), zOffset * childWidth.z());
		V3d childEnd = childBeg + childWidth;
		PV3d childBoundary = { childBeg, childEnd };

		node->childs[i] = new OctreeNode(nAllNodes++, depth + 1, childWidth, childBoundary, childPointsIdx[i]);
		node->childs[i]->parent = node;

		createNode(node->childs[i], depth + 1, childWidth, childBoundary, modelVerts, childPointsIdx[i]);
	}
}

// used to construct b-spline function
inline std::tuple<vector<PV3d>, vector<size_t>> Octree::setInDomainPoints(OctreeNode* node, const int& cornerID, map<size_t, bool>& visID)
{
	auto temp = node->childs[cornerID];
	vector<PV3d> points;
	vector<size_t> pointsID;

	while (temp->depth + 1 != maxDepth)
	{
		visID[temp->id] = true;
		points.emplace_back(std::make_pair(temp->corners[cornerID], temp->width));
		pointsID.emplace_back((temp->id) * 8 + cornerID);
		if (temp->isLeaf) break;
		temp = temp->childs[cornerID];
	}

	return std::make_tuple(points, pointsID);
}

//////////////////////
//    Save  Data    //
//////////////////////
inline void Octree::saveDomain2OBJFile(const string& filename) const
{
	checkDir(filename);
	std::ofstream out(filename);
	if (!out) { fprintf(stderr, "IO Error: File %s could not be opened!", filename.c_str()); return; }

	cout << "domain Nodes size = " << inDmNodes[0].size() << endl;
	int count = -8;
	for (const auto& leaf : inDmNodes[0])
	{
		for (int i = 0; i < 8; i++)
		{
			out << "v " << std::setiosflags(std::ios::fixed) << std::setprecision(9) << leaf->corners[i].x() << " " << leaf->corners[i].y() << " " << leaf->corners[i].z() << endl;
			count++;
		}
		out << "l " << 1 + count << " " << 2 + count << endl;
		out << "l " << 1 + count << " " << 3 + count << endl;
		out << "l " << 1 + count << " " << 5 + count << endl;
		out << "l " << 2 + count << " " << 4 + count << endl;
		out << "l " << 2 + count << " " << 6 + count << endl;
		out << "l " << 3 + count << " " << 4 + count << endl;
		out << "l " << 3 + count << " " << 7 + count << endl;
		out << "l " << 4 + count << " " << 8 + count << endl;
		out << "l " << 5 + count << " " << 6 + count << endl;
		out << "l " << 5 + count << " " << 7 + count << endl;
		out << "l " << 6 + count << " " << 8 + count << endl;
		out << "l " << 7 + count << " " << 8 + count << endl;
	}
	out.close();
}

inline void Octree::saveNodeCorners2OBJFile(const string& filename) const
{
	checkDir(filename);
	std::ofstream out(filename);
	if (!out) { fprintf(stderr, "IO Error: File %s could not be opened!", filename.c_str()); return; }

	cout << "leafNodes size = " << leafNodes.size() << endl;
	int count = -8;
	for (const auto& leaf : leafNodes)
	{
		for (int i = 0; i < 8; i++)
		{
			out << "v " << std::setiosflags(std::ios::fixed) << std::setprecision(9) << leaf->corners[i].x() << " " << leaf->corners[i].y() << " " << leaf->corners[i].z() << endl;
			count++;
		}
		out << "l " << 1 + count << " " << 2 + count << endl;
		out << "l " << 1 + count << " " << 3 + count << endl;
		out << "l " << 1 + count << " " << 5 + count << endl;
		out << "l " << 2 + count << " " << 4 + count << endl;
		out << "l " << 2 + count << " " << 6 + count << endl;
		out << "l " << 3 + count << " " << 4 + count << endl;
		out << "l " << 3 + count << " " << 7 + count << endl;
		out << "l " << 4 + count << " " << 8 + count << endl;
		out << "l " << 5 + count << " " << 6 + count << endl;
		out << "l " << 5 + count << " " << 7 + count << endl;
		out << "l " << 6 + count << " " << 8 + count << endl;
		out << "l " << 7 + count << " " << 8 + count << endl;
	}
	out.close();
}