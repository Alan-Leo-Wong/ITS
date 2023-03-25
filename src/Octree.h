#pragma once
#include "BaseModel.h"
#include "utils\String.hpp"
#include "utils\Geometry.hpp"

struct OctreeNode
{
public:
	long int id;

	int depth;
	bool isLeaf;
	bool isInterMesh; // if intersect with mesh

	V3d width;
	V3d corners[8];
	PV3d boundary;  // <back/bottom/left, front/top/right>	
	vector<PV3d> edges; // used to compute intersection between mesh's faces and node's edges
	vector<size_t> idxOfPoints; // index of points 

	OctreeNode* parent;
	vector<OctreeNode*> childs;

	std::array<V3d, 2> domain; // influence domain

public:
	OctreeNode()
	{
		parent = nullptr;
		depth = 0;
		width = V3d(.0, .0, .0);
		isLeaf = true;
		isInterMesh = false;
	}

	// 虚拟节点
	OctreeNode(const int& _id, const int& _depth, const V3d& _width, const PV3d& _boundary)
	{
		id = _id;
		parent = nullptr;
		depth = _depth;
		width = _width;
		boundary = _boundary;
		isLeaf = true;
		isInterMesh = false;
	}

	OctreeNode(const int& _id, const int& _depth, const V3d& _width, const PV3d& _boundary, const std::vector<size_t>& _idxOfPoints)
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

	bool isInDomain(const PV3d& q_boundary); // whether in otherNode's domain
};

class Octree : public BaseModel
{
protected:
	int maxDepth = -1;
	int numNodes = 0;
	int nLeafNodes = 0;
	double scaleSize = 0.01;
	string modelName;

	VXd sdfVal;

	OctreeNode* root;
	BoundingBox bb;

	vector<OctreeNode*> leafNodes;
	vector<OctreeNode*> allNodes;

	vector<OctreeNode*> interLeafNodes;
	vector<vector<OctreeNode*>> inDmNodes;

	vector<PV3d> nodeXEdges; // 所有节点X轴方向的边，只用于求交
	vector<PV3d> nodeYEdges; // 所有节点Y轴方向的边，只用于求交
	vector<PV3d> nodeZEdges; // 所有节点Z轴方向的边，只用于求交

	VXd BSplineValue;
	vector<V3d> interPoints; // intersection points

	VXd lambda;

	map<V3d, vector<PUII>> corner2IDs;

	double minBVal, maxBVal;

public:
	// constructor and destructor
	Octree(const int& _maxDepth, const string& _modelPath, const double& _scaleSize = 0.1) :
		maxDepth(_maxDepth), modelName(getFileName(DELIMITER, _modelPath)), scaleSize(_scaleSize)
	{
		readFile(_modelPath);
		createOctree(scaleSize);
	}

	~Octree() { delete root; root = nullptr; };

public:
	void createOctree(const double& scaleSize);

	void createNode(OctreeNode*& node, const int& depth, const V3d& width, const std::pair<V3d, V3d>& boundary, const vector<size_t>& idxOfPoints);

	void cpIntersection();

	void cpCoefficients();

	std::tuple<vector<PV3d>, vector<size_t>> setInDomainPoints(OctreeNode* node, map<size_t, bool>& visID);
	std::tuple<vector<PV3d>, vector<size_t>> setInDomainPoints(OctreeNode* node, const int& cornerID, map<size_t, bool>& visID);

	void setSDF();

	void setBSplineValue();

public:
	// save data
	void saveDomain2OBJFile(const string& filename) const;

	void saveIntersections(const string& filename, const vector<V3d>& intersections) const;

	void saveNodeCorners2OBJFile(const string& filename) const;

	void saveSDFValue(const string& filename) const;

	void saveCoefficients(const string& filename) const;

	void saveBSplineValue(const string& filename) const;

	// visulization
	void mcVisualization(const string& filename, const V3i& resolution, const double& isoVal = .0) const;

	void textureVisualization(const string& filename) const;
};