#pragma once
#include "MyBaseModel.h"

inline double BaseFunction(double x, double width, double node_x);

inline double dBaseFunction(double x, double width, double node_x);

struct OctreeNode
{
	OctreeNode* parent;
	vector<OctreeNode*> childs;

	int depth;   //�������
	V3d width;
	V3d corners[8];
	bool isLeaf;
	bool isIntersectWithMesh;
	double SDFValue[8];

	PV3d boundary;  // ���ӱ߽�, <back/bottom/left, front/top/right>	
	vector<size_t> idxOfPoints; // �����ĵ��index

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
};

class MyOctree : public MyBaseModel
{
protected:
	int maxDepth;

	OctreeNode* root;

	MXd nodeVerts;    // �洢Ҷ�ӽڵ�

	vector<vector<PV3d>> nodeXEdges; // �ڵ�X�᷽��ıߣ�ֻ������
	vector<vector<PV3d>> nodeYEdges; // �ڵ�Y�᷽��ıߣ�ֻ������
	vector<vector<PV3d>> nodeZEdges; // �ڵ�Z�᷽��ıߣ�ֻ������

	vector<OctreeNode*> leafNodes;
	vector<V3d> intersections;

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

	vector<OctreeNode*> getLeafNodes();

	void saveNodeCorners2OBJFile(const string& filename);

	void cpIntersection();

	void createNode(OctreeNode*& node, const int& depth, const V3d& width, const std::pair<V3d, V3d>& boundary, const vector<size_t>& idxOfPoints);

	void saveIntersections(const string& filename, const vector<V3d>& intersections) const;

	SpMat coEfficientOfPoints(const vector<V3d>& edgeIntersections, const vector<V3d>& faceIntersections, SpMat& Bx, SpMat& By, SpMat& Bz);

	void saveBValue2TXT(const string& filename, const Eigen::VectorXd& X) const;
};