#pragma once
#include "BaseModel.h"
#include <unordered_map>

struct OctreeNode
{
public:
	size_t id;

	int depth;
	V3d width;
	V3d corners[8];
	double sdf[8];

	PV3d boundary;            // <back/bottom/left, front/top/right>	
	vector<PV3d> edges;       // used to compute intersection between mesh's faces and node's edges
	vector<uint> idxOfPoints; // index of points 

	bool isLeaf;
	bool isInterMesh;         // if intersect with mesh

	OctreeNode* parent;
	vector<OctreeNode*> childs;
	//std::array<V3d, 2> domain; // influence domain

public:
	OctreeNode()
	{
		parent = nullptr;
		depth = 0;
		width = V3d(.0, .0, .0);
		isLeaf = true;
		isInterMesh = false;
	}

	OctreeNode(const size_t& _id, const int& _depth, const V3d& _width, const PV3d& _boundary)
	{
		id = _id;
		parent = nullptr;
		depth = _depth;
		width = _width;
		boundary = _boundary;
		isLeaf = true;
		isInterMesh = false;
	}
	
	OctreeNode(const size_t& _id, const int& _depth, const V3d& _width, const PV3d& _boundary, const std::vector<uint>& _idxOfPoints)
	{
		id = _id;
		parent = nullptr;
		depth = _depth;
		width = _width;
		boundary = _boundary;
		idxOfPoints = _idxOfPoints;
		isLeaf = true;
		isInterMesh = false;
	}

	~OctreeNode()
	{
		/*delete parent;
		parent = nullptr;

		for (int i = 0; i < 8; ++i)
		{
			delete child[i];
			child[i] = nullptr;
		}
		delete[] child;*/
	}

	void setEdges();

	void setCorners();

	void setCornersIdx(map<V3d, vector<PUII>>& corner2IDs);
	void setCornersIdx(map<V3d, vector<PUII>>& corner2IDs, const uint& _id);

	//double BaseFunction4Point(const V3d& p);

	//bool isInDomain(const PV3d& q_boundary); // whether in otherNode's domain
};

#define GET_CYCLE_BIT(x, n, N)  ((((x) >> (N - n)) | ((x) << (n))) & ((1 << (N)) - 1)) // 得到N位二进制数左移n位后的循环码: (x >> (N - n)) | (x << n)
#define GET_CHILD_ID(x, y)	    (((x) << 3) + (y >= (0x4) ? ((0x4) | GET_CYCLE_BIT(y - (0x4), 1, 2)) : (GET_CYCLE_BIT(y, 1, 2))) + 1) // x为parent, y为offset(0<=x<=7)
#define GET_OFFSET(x)           ((x - 1) & (0x7))			 // 得到节点x(x>=1)的最后三位，代表x在父节点中的偏移位置
/*
* 000 - bottom back  left
* 111 - up     front right
*/
#define GET_PARENT_ID(x)        ((x - 1) >> 3)				 // 得到已有的节点x找到对应的父节点(x>=1)
class Octree
{
	friend class ThinShells;
private:
	OctreeNode* root;
	V3d treeOrigin;

	int maxDepth = -1;
	uint nAllNodes = 0;
	uint nLeafNodes = 0;

	vector<OctreeNode*> leafNodes;
	vector<OctreeNode*> allNodes;
	vector<vector<OctreeNode*>> inDmNodes;

	vector<OctreeNode*> d_leafNodes; // 存储最深层的且穿过表面的叶子节点

	map<V3d, vector<PUII>> corner2IDs; // 这里的id与建立自适应表面的八叉树使用的id不同，它的id等于当前建的格子数，从0开始

	std::unordered_map<size_t, int> visNodeId;
	// 得到id后构建对应的node，后期将改成仅用于可视化——因为其实实际上只要让d_leafNodes存储所有表面附近的叶子节点id，
	 // 然后得到对应node的坐标和大小（这时候id2Node这个map就需要改成id to {coord,width}的映射了），就可以定义基函数了
	std::unordered_map<size_t, OctreeNode*> id2Node; 

private:
	void createNode(OctreeNode*& node, const int& depth,
		const V3d& width, const std::pair<V3d, V3d>& boundary,
		const vector<V3d> modelVerts, const vector<uint>& idxOfPoints);

	V3d getNodeCoord(const size_t& nodeId, const V3d& width);

	std::tuple<vector<PV3d>, vector<size_t>> setInDomainPoints(OctreeNode* node, map<size_t, bool>& visID);

public:
	// constructor and destructor
	Octree() { root = nullptr; }

	Octree(const int& _maxDepth, const BoundingBox& bb,
		const uint& nPoints, const vector<V3d>& modelVerts) : maxDepth(_maxDepth)
	{
		createOctree(bb, nPoints, modelVerts);
	}

	~Octree() { delete root; root = nullptr; };

public:
	void createOctree(const BoundingBox& bb, const uint& nPoints, const vector<V3d>& modelVerts);

public:
	// save data
	void saveDomain2OBJFile(const string& filename) const;

	void saveNodeCorners2OBJFile(const string& filename) const;
};