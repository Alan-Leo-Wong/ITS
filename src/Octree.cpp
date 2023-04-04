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
	std::iota(idxOfPoints.begin(), idxOfPoints.end(), 0); // 0 ~ nPoints-1

	V3d width = bb.boxWidth;
	PV3d boundary = std::make_pair(bb.boxOrigin, bb.boxEnd);
	treeOrigin = bb.boxOrigin;

	root = new OctreeNode(0, 0, width, boundary, idxOfPoints);

	createNode(root, 0, width, boundary, modelVerts, idxOfPoints);

	if (!d_leafNodes.empty() && maxDepth > 2) createSurfaceNode(d_leafNodes[0]->width);
}

#define GET_CYCLE_BIT(x, n, N)  ((((x) >> (N - n)) | ((x) << (n))) & ((1 << (N)) - 1)) // 得到N位二进制数左移n位后的循环码: (x >> (N - n)) | (x << n)
#define GET_CHILD_ID(x, y)	    (((x) << 3) + (y >= (0x4) ? ((0x4) | GET_CYCLE_BIT(y - (0x4), 1, 2)) : (GET_CYCLE_BIT(y, 1, 2))) + 1) // x为parent, y为offset(0<=x<=7)

inline void Octree::createNode(OctreeNode*& node, const int& depth,
	const V3d& width, const std::pair<V3d, V3d>& boundary,
	const vector<V3d> modelVerts, const vector<uint>& idxOfPoints)
{
	id2Node[node->id] = node;

	allNodes.emplace_back(node);
	nAllNodes++;

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
	const size_t id = node->id;
	for (int i = 0; i < 8; i++)
	{
		const int xOffset = i & 1;
		const int yOffset = (i >> 1) & 1;
		const int zOffset = (i >> 2) & 1;
		V3d childBeg = b_beg + V3d(xOffset * childWidth.x(), yOffset * childWidth.y(), zOffset * childWidth.z());
		V3d childEnd = childBeg + childWidth;
		PV3d childBoundary = { childBeg, childEnd };

		node->childs[i] = new OctreeNode(GET_CHILD_ID(id, i), depth + 1, childWidth, childBoundary, childPointsIdx[i]);
		node->childs[i]->parent = node;

		createNode(node->childs[i], depth + 1, childWidth, childBoundary, modelVerts, childPointsIdx[i]);
	}
}

#define GET_OFFSET(x)                ((x - 1) & (0x7))			 // 得到节点x(x>=1)的最后三位，代表x在父节点中的偏移位置
/*
* 000 - bottom back  left
* 111 - up     front right
*/
#define GET_PARENT_ID(x)             ((x - 1) >> 3)				 // 得到已有的节点x找到对应的父节点(x>=1)

#define GET_CROSS_EDGE_VAL_0(subDepth)  ()
#define GET_CROSS_EDGE_VAL_1(subDepth)  ()
#define GET_CROSS_EDGE_VAL_2(subDepth)  ()

// 建立模型表面的格子
inline void Octree::createSurfaceNode(const V3d& leafWidth)
{
	std::queue<size_t> q;
	for (auto node : d_leafNodes)
		q.push(node->id);

	enum OFFSET
	{
		BOTTOM_BACK_LEFT,
		BOTTOM_BACK_RIGHT,
		BOTTOM_FRONT_LEFT,
		BOTTOM_FRONT_RIGHT,
		UPPER_BACK_LEFT,
		UPPER_BACK_RIGHT,
		UPPER_FRONT_LEFT,
		UPPER_FRONT_RIGHT
	};

	enum CROSS_EDGE
	{
		TO_RIGHT,
		TO_LEFT,

		TO_DOWN,
		TO_UP,

		TO_FRONT,
		TO_BACK,
	};

	auto getCrossEdgeVal = [=](const int& i, const int& crossEdgeKind, const int& subDepth)->long long
	{
		long long val = (crossEdgeKind & 1) ? -1 : 1;
		if (i == 0) val *= GET_CROSS_EDGE_VAL_0(subDepth);
		else if (i == 1) val *= GET_CROSS_EDGE_VAL_1(subDepth);
		else if (i == 2) val *= GET_CROSS_EDGE_VAL_2(subDepth);
		return val;
	};

	auto getSurfaceNodeId = [&](const int& offset, const size_t& queryNodeId, size_t needNodeId[3])
	{
		int t_offset = offset;

		int fatherId = GET_PARENT_ID(queryNodeId);
		if (!fatherId) return; // 保证queryNode至少位于第2层及以下
		int fatherOffset = GET_OFFSET(fatherId);
		for (int i = 0; i < 3; ++i, t_offset >>= 1)
		{
			int queryBit = ~(t_offset & 1);
			int t_fatherId = fatherId, t_fatherOffset = fatherOffset;
			int subDepth = 0;

			// 当queryNode的某个父节点fatherNode在其父节点中的偏移量fatherOffset的第i位与queryBit(表示了queryNode待寻找的周围节点的反方向)一致时
			// 我们就可以计算待寻找的queryNode周围节点id了
			while ((t_fatherOffset >> i & 1) != queryBit)
			{
				t_fatherId = GET_PARENT_ID(t_fatherId);
				if (!t_fatherId) { subDepth = 0; break; }

				t_fatherOffset = GET_OFFSET(t_fatherId);
				++subDepth;
			}

			CROSS_EDGE kind = CROSS_EDGE(i * 2 + queryBit);
			if (subDepth >= 1) needNodeId[i] = getCrossEdgeVal(i, kind, subDepth);
		}
	};

	const size_t stNodeId = std::pow(8, maxDepth - 2);
	// 判断是否穿过表面
	auto isCrossSurface = [=](const int& nodeId, const V3d& width)->bool
	{
		// 得到nodeId对应的八个顶点
		/*const size_t nodeParentOffset = ((nodeId - stNodeId) & 0x000) + 1;
		const size_t x_offset = ;*/
	};

	// 沿表面扩散
	while (!q.empty())
	{
		auto queryNodeId = q.front();
		q.pop();

		if (visNodeId[queryNodeId]) continue;
		visNodeId[queryNodeId] = true;

		// 得到其在父亲内是哪一个偏移节点
		OFFSET offset = (OFFSET)GET_OFFSET(queryNodeId);
		// 得到需要建立的三个nodeId(通过找父亲以及定值)
		size_t needNodeId[3] = { 0 };
		getSurfaceNodeId(offset, queryNodeId, needNodeId);

		// 判断是否建立过且sdf值有正有负
		for (const auto nodeId : needNodeId)
		{
			if (visNodeId[nodeId] && !isCrossSurface(nodeId, leafWidth)) continue;

			// create node
			vector<size_t> parents;
			int parentNodeId = GET_PARENT_ID(nodeId);
			parents.emplace_back(parentNodeId); // 逐渐向上遍历树，找到所有父节点
			while (id2Node[parentNodeId] == nullptr)
			{
				parentNodeId = GET_PARENT_ID(nodeId);
				parents.emplace_back(parentNodeId);
			}

			for (size_t j = parents.size() - 1; j >= 0; --j)
			{
				const size_t parentId = parents[j];
				OctreeNode* node = id2Node[parentId];

				const V3d b_beg = node->boundary.first;
				const V3d b_end = node->boundary.second;
				const V3d childWidth = (b_end - b_beg) * 0.5;
				const int depth = node->depth;
				const size_t id = node->id;

				for (int i = 0; i < 8; i++)
				{
					const int xOffset = i & 1;
					const int yOffset = (i >> 1) & 1;
					const int zOffset = (i >> 2) & 1;
					V3d childBeg = b_beg + V3d(xOffset * childWidth.x(), yOffset * childWidth.y(), zOffset * childWidth.z());
					V3d childEnd = childBeg + childWidth;
					PV3d childBoundary = { childBeg, childEnd };

					const int childId = GET_CHILD_ID(id, i);
					node->childs[i] = new OctreeNode(childId, depth + 1, childWidth, childBoundary);
					node->childs[i]->parent = node;
					id2Node[childId] = node->childs[i];
				}
			}

			q.push(nodeId);
		}
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