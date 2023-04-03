#include "Octree.h"
#include "utils\String.hpp"
#include <numeric>
#include <iomanip>
#include <queue>

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
void Octree::createOctree(const BoundingBox& bb, const uint& nPoints, const vector<V3d>& modelVerts)
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
		if (depth + 1 >= maxDepth) d_leafNodes.emplace_back(node);
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

/* used to construct b - spline function */
std::tuple<vector<PV3d>, vector<size_t>> Octree::setInDomainPoints(OctreeNode* node, map<size_t, bool>& visID)
{
	auto temp = node->parent;
	vector<PV3d> points;
	vector<size_t> pointsID;

	auto getCorners = [&](OctreeNode* node)
	{
		for (int k = 0; k < 8; ++k)
		{
			V3d i_corner = node->corners[k];
			for (const auto& id_ck : corner2IDs[i_corner])
			{
				const uint o_id = id_ck.first;
				const uint o_k = id_ck.second;
				const uint o_realID = o_id * 8 + o_k;

				if (visID[o_realID] && fabs(allNodes[o_id]->width[0] - node->width[0]) > 1e-9) continue;
				visID[o_realID] = true;

				points.emplace_back(std::make_pair(i_corner, allNodes[o_id]->width));
				pointsID.emplace_back(o_realID);
			}
		}
	};

	while (temp != nullptr)
	{
		getCorners(temp);
		temp = temp->parent;
	}

	return std::make_tuple(points, pointsID);
}

// 建立模型表面的格子
void Octree::createSurfaceNode(OctreeNode*& node)
{
	std::queue<size_t> q;
	for (auto node : d_leafNodes)
		q.push(node->id);

	enum OFFSET
	{
		LEFT_BOTTOM_BACK,
		LEFT_BOTTOM_FRONT,
		RIGHT_BOTTOM_BACK,
		RIGHT_BOTTOM_FRONT,
		LEFT_UPPER_BACK,
		LEFT_UPPER_FRONT,
		RIGHT_UPPER_BACK,
		RIGHT_UPPER_FRONT,
	};

	// 沿表面扩散
	while (!q.empty())
	{
		auto queryNodeId = q.front();
		q.pop();

		if (visNodeId[queryNodeId]) continue;

		// 得到其在父亲内是哪一个偏移节点
		OFFSET offset = (OFFSET)getOffset(queryNodeId);
		// 得到需要建立的三个nodeId(通过找父亲以及定值)
		int needNodeId[3] = { -1 };
		getSurfaceNodeId(needNodeId);
		switch (offset)
		{
		case LEFT_BOTTOM_BACK:
			break;

		case LEFT_BOTTOM_FRONT:
			break;

		case RIGHT_BOTTOM_BACK:
			break;

		case RIGHT_BOTTOM_FRONT:
			break;

		case LEFT_UPPER_BACK:
			break;

		case LEFT_UPPER_FRONT:
			break;

		case RIGHT_UPPER_BACK:
			break;

		case RIGHT_UPPER_FRONT:
			break;
		}

		// 判断是否建立过
		for (const auto nodeId : needNodeId)
		{
			if (!visNodeId[nodeId])
			{
				// create node
			}
		}
	}
}

//////////////////////
//    Save  Data    //
//////////////////////
void Octree::saveDomain2OBJFile(const string& filename) const
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

Octree& Octree::operator=(const Octree& tree)
{
	cout << "Execute Copy Assignment operator...\n";
	root = new OctreeNode(*(tree.root)); // deep copy

	maxDepth = tree.maxDepth;
	nAllNodes = tree.nAllNodes;
	nLeafNodes = tree.nLeafNodes;

	leafNodes.resize(nLeafNodes);
	for (int i = 0; i < nLeafNodes; ++i)
	{
		leafNodes[i] = new OctreeNode(*(tree.leafNodes[i]));
		if (tree.leafNodes[i]->parent != nullptr)
			leafNodes[i]->parent = new OctreeNode(*(tree.leafNodes[i]->parent));
	}

	allNodes.resize(nAllNodes);
	for (int i = 0; i < nAllNodes; ++i)
	{
		allNodes[i] = new OctreeNode(*(tree.allNodes[i]));
		if (tree.allNodes[i]->parent != nullptr)
			allNodes[i]->parent = new OctreeNode(*(tree.allNodes[i]->parent));
	}

	corner2IDs = tree.corner2IDs;

	return *this;
}

void Octree::saveNodeCorners2OBJFile(const string& filename) const
{
	checkDir(filename);
	std::ofstream out(filename);
	if (!out) { fprintf(stderr, "IO Error: File %s could not be opened!", filename.c_str()); return; }

	//cout << "leafNodes size = " << leafNodes.size() << endl;
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