#pragma once
#include "BaseModel.h"

struct OctreeNode
{
public:
	uint id;

	int depth;
	V3d width;
	V3d corners[8];
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

	OctreeNode(const int& _id, const int& _depth, const V3d& _width, const PV3d& _boundary, const std::vector<uint>& _idxOfPoints)
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

	//double BaseFunction4Point(const V3d& p);

	//bool isInDomain(const PV3d& q_boundary); // whether in otherNode's domain
};

class Octree
{
	friend class ThinShells;
private:
	OctreeNode* root;

	int maxDepth = -1;
	uint nAllNodes = 0;
	uint nLeafNodes = 0;

	vector<OctreeNode*> leafNodes;
	vector<OctreeNode*> allNodes;
	vector<vector<OctreeNode*>> inDmNodes;

	map<V3d, vector<PUII>> corner2IDs;

public:
	// constructor and destructor
	Octree() {}

	Octree(const int& _maxDepth, const BoundingBox& bb,
		const uint& nPoints, const vector<V3d>& modelVerts) : maxDepth(_maxDepth)
	{
		createOctree(bb, nPoints, modelVerts);
	}

	~Octree() { delete root; root = nullptr; };

public:
	void createOctree(const BoundingBox& bb, const uint& nPoints, const vector<V3d>& modelVerts);

	void createNode(OctreeNode*& node, const int& depth,
		const V3d& width, const std::pair<V3d, V3d>& boundary,
		const vector<V3d> modelVerts, const vector<uint>& idxOfPoints);

	std::tuple<vector<PV3d>, vector<size_t>> setInDomainPoints(OctreeNode* node, map<size_t, bool>& visID);

	Octree& operator=(const Octree&);

public:
	// save data
	void saveDomain2OBJFile(const string& filename) const;

	void saveNodeCorners2OBJFile(const string& filename) const;
};