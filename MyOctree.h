#pragma once
#include "MyBaseModel.h"

inline double BaseFunction(double x, double width, double node_x);

inline double dBaseFunction(double x, double width, double node_x);

struct OctreeNode
{
	OctreeNode* parent;
	OctreeNode* child[8];

	int depth;   //格子深度
	V3d width;
	V3d corners[8];
	bool isLeaf;
	bool isIntersectWithMesh;
	double SDFValue[8];

	std::pair<V3d, V3d> boundary;  // 格子边界, <back/bottom/left, front/top/right>	
	std::vector<size_t> idxOfPoints; // 包含的点的index

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

		for (int i = 0; i < 8; ++i)
			child[i] = new OctreeNode;
	}

	~OctreeNode()
	{
		delete parent;
		delete[] child;

		for (int i = 0; i < 8; ++i)
			child[i] = nullptr;
	}
};

class MyOctree : public MyBaseModel
{
protected:
	int maxDepth;

	OctreeNode* root;

	MXd nodeVerts;    // 存储叶子节点

	vector<vector<PV3d>> nodeXEdges; // 节点X轴方向的边，只用于求交
	vector<vector<PV3d>> nodeYEdges; // 节点Y轴方向的边，只用于求交
	vector<vector<PV3d>> nodeZEdges; // 节点Z轴方向的边，只用于求交

	vector<OctreeNode*> leafNodes;

	string modelName;

public:
	MyOctree(int _maxDepth, string _modelName) :maxDepth(_maxDepth), modelName(_modelName) 
	{ 
		root = new OctreeNode;
		nodeXEdges.resize(maxDepth);
		nodeYEdges.resize(maxDepth);
		nodeZEdges.resize(maxDepth);
	}

	~MyOctree() { delete root; root = nullptr; };

	void createOctree(const double& scaleSize = 0.1);

	void selectLeafNode(OctreeNode* node);

	void saveNodeCorners2OBJFile(string filename); 
	
	void cpIntersection(vector<V3d>& edgeIntersections, vector<V3d>& facetIntersections, vector<V3d>& edgeUnitNormals, vector<V3d>& facetUnitNormals);

	void createNode(OctreeNode*& node, OctreeNode* parent, const int& depth, const V3d& width, const vector<size_t>& idxOfPoints, const std::pair<V3d, V3d>& boundary);

	//void Intersection(vector<V3d>& edgeIntersections, vector<V3d>& facetIntersections, vector<V3d>& edgeUnitNormals, vector<V3d>& facetUnitNormals);
	//void SaveIntersections(string filename, vector<V3d> intersections, vector<V3d>& UnitNormals) const;
	///*void*/ SaveBValue2TXT(string filename, Eigen::VectorXd X) const;
	//SpMat CeofficientOfPoints(const vector<V3d> edgeIntersections, const vector<V3d> faceIntersections, SpMat& Ax, SpMat& Ay, SpMat& Az); // p = A[p](lambda_0, ..., lambda_n)^T
	//Eigen::VectorXd Solver(Eigen::VectorXd &VertexValue, const double alpha);
	vector<OctreeNode*> getLeafNodes();
};