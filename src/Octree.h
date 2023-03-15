#pragma once
#include "BaseModel.h"

double BaseFunction(const double& x, const double& node_x, const double& w);

struct OctreeNode
{
	OctreeNode* parent;
	vector<OctreeNode*> childs;

	int depth;   //格子深度
	V3d width;
	V3d corners[8];
	bool isLeaf;
	bool isIntersectWithMesh;
	double SDFValue[8];

	vector<PV3d> edges;

	PV3d boundary;  // 格子边界, <back/bottom/left, front/top/right>	
	vector<size_t> idxOfPoints; // 包含的点的index

	std::array<V3d, 4> domain; // 影响范围

	OctreeNode()
	{
		parent = nullptr;
		depth = 0;
		width = V3d(.0, .0, .0);
		isLeaf = true;
		isIntersectWithMesh = false;
	}

	OctreeNode(const int& _depth, const V3d& _width, const std::pair<V3d, V3d> _boundary, const std::vector<size_t>& _idxOfPoints)
	{
		parent = nullptr;
		depth = _depth;
		width = _width;
		boundary = _boundary;
		idxOfPoints = _idxOfPoints;
		isLeaf = true;
		isIntersectWithMesh = false;
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

	void setCorner();

	void setEdges();

	void setDomain(); // 设置影响范围

	bool isInDomain(const OctreeNode* node);

	inline double BaseFunction4Point(const V3d& p);
};

class Octree : public BaseModel
{
protected:
	int maxDepth;
	int numNodes = 0;
	int numLeafNodes = 0;

	OctreeNode* root;

	MXd nodeVerts;    // 存储叶子节点

	std::unordered_map<int, OctreeNode*> edgeIdx2Node;
	vector<PV3d> nodeXEdges; // 节点X轴方向的边，只用于求交
	vector<PV3d> nodeYEdges; // 节点Y轴方向的边，只用于求交
	vector<PV3d> nodeZEdges; // 节点Z轴方向的边，只用于求交

	vector<OctreeNode*> leafNodes;
	vector<OctreeNode*> intersectLeafNodes;
	vector<vector<OctreeNode*>> inDomainLeafNodess;

	vector<V3d> intersections;
	Eigen::VectorXd BSplineValue;

	string modelName;

public:
	Octree(int _maxDepth, string _modelName) :maxDepth(_maxDepth), modelName(_modelName) { root = new OctreeNode; }

	~Octree() { delete root; root = nullptr; };

	void createOctree(const double& scaleSize = 0.1);

	//void selectLeafNode(OctreeNode* node);

	vector<OctreeNode*> getLeafNodes();

	void saveNodeCorners2OBJFile(const string& filename);

	void cpIntersection();

	void createNode(OctreeNode*& node, const int& depth, const V3d& width, const std::pair<V3d, V3d>& boundary, const vector<size_t>& idxOfPoints);

	void saveIntersections(const string& filename, const vector<V3d>& intersections) const;

	void setInDomainLeafNode();

	void setSDF();

	void setBSplineValue();

	void saveBValue(const string& filename, const Eigen::VectorXd& X) const;
};