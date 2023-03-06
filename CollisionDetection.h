#pragma once
#include "ThinShells.h"

class CollisionDetection
{
public:
	
	CollisionDetection() {};
	~CollisionDetection() {};
	// compute the intersection of two cuboids
	vector<V3d> IntersectionOfCuboid(const vector<V3d>& cube1, const vector<V3d>& cube2);
	// initialize the objects
	void Initialize(const int& resolution1, const int& resolution2, const string& modelName1, const string& modelName2, 
		const string& format1, const string& format2);
	// compute B-spline basis with unvariable
	double BaseFunction(const double& x, const double& node_x, const double& w);
	double BaseFunction4Point(const V3d& p, const V3d& nodePosition, const V3d& width);
	// Compute the spline value in a specified node field
	double ComputeSplineValue(const ThinShells& model, const V3d& p, const V3d& minBound);
	// Extract the nodes that both 0-level IBSs pass through 
	void ExtractCommonNodes(vector<V3d>& commonNodesVertex, vector<vector<int>>& commonNodesID, vector<double>& value1, vector<double>& value2);
	vector<std::pair<V3d, V3d>> ExtractInterLinesInSingleNode(const vector<V3d>& verts, const vector<int>& node, const vector<double>& val1, const vector<double>& val2);
	void ExtractIntersectLines(const int& resolution1, const int& resolution2, const string& modelName1,
		const string& modelName2, const string& format1, const string& format2);
	void SaveNode(const string& filename, const vector<vector<int>>& overlappingNodesID , const MXd& nodes) const;
	void SaveNode(const string& filename, const vector<vector<int>>& overlappingNodesID , const vector<V3d>& nodes) const;
protected:
	const int m_nodeEdges[12][2] = {
		{0, 1}, {3, 2}, {4, 5}, {7, 6}, 
		{0, 3}, {1, 2}, {4, 7}, {5, 6}, 
		{0, 4}, {1, 5}, {3, 7}, {2, 6}
	};
	const int m_faceEdge[6][4] = {
		{0, 5, 1, 4}, {2, 7, 3, 6}, {1, 10, 3, 11}, {0, 9, 2, 8}, {9, 5, 11, 7}, {10, 4, 8, 6}
	};
	const int m_edgeFace[12][2] = {
		{0, 3}, {0, 2}, {1, 3}, {1, 2},
		{0, 5}, {0, 4}, {1, 5}, {1, 4},
		{3, 5}, {3, 4}, {2, 5}, {2, 4}
	};
	ThinShells m_model1;     
	ThinShells m_model2;
	bool m_is_collision;
};

