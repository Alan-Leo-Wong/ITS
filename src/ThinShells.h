#pragma once
#include "SVO.h"
#include "BaseModel.h"
#include "utils\String.hpp"

class ThinShells : public BaseModel
{
private:
	V3i svo_gridSize;
	SparseVoxelOctree svo;
	//vector<V3d> nodeWidthArray;

	vector<V3d> edgeInterPoints; // Intersection points of octree node and mesh's edges
	vector<V3d> faceInterPoints; // Intersection points of octree node's edges and mesh's faces
	vector<V3d> allInterPoints;  // All intersection points of octree node and mesh

private:
	VXd sdfVal;
	VXd lambda;
	VXd bSplineVal;

private:
	double innerShellIsoVal = -DINF;
	double outerShellIsoVal = -DINF;

public:
	int treeDepth;

public:
	// constructor and destructor
	ThinShells() {}

	ThinShells(const string& filename, const int& _grid_x, const int& _grid_y, const int& _grid_z) :
		svo_gridSize(_grid_x, _grid_y, _grid_z), BaseModel(filename), svo(_grid_x, _grid_y, _grid_z)
	{
		svo.createOctree(nModelTris, modelTris, modelBoundingBox, concatFilePath((string)VIS_DIR, modelName));
		treeDepth = svo.treeDepth;
#ifndef NDEBUG
		saveTree("");
#endif // !NDEBUG
	}

	ThinShells(const string& filename, const V3i& _grid) :svo_gridSize(_grid), BaseModel(filename), svo(_grid)
	{
		svo.createOctree(nModelTris, modelTris, modelBoundingBox, concatFilePath((string)VIS_DIR, modelName));
		treeDepth = svo.treeDepth;
#ifndef NDEBUG
		saveTree("");
#endif // !NDEBUG
	}

	~ThinShells() {}

	// ThinShells& operator=(const ThinShells& model);

private:
	//void cpIntersectionPoints();
	void cpIntersectionPoints();

	void cpSDFOfTreeNodes();

	void cpCoefficients();

	void cpBSplineValue();

	void initBSplineTree();

	void setLatentMatrix(const double& alpha);

public:
	void creatShell();

	// Octree& bSplineTree() { return bSplineTree; }
	// const Octree& bSplineTree() const { return bSplineTree; }

	std::array<double, 2> getShellIsoVal() { return { innerShellIsoVal, outerShellIsoVal }; }

	// 点在表面的查询
	void singlePointQuery(const std::string& out_file, const V3d& point);

	void multiPointQuery(const std::string& out_file, const vector<V3d>& points);

	void multiPointQuery(const std::string& out_file, const MXd& points);

public:
	void saveTree(const string& filename) const;

	void saveIntersections(const string& filename, const vector<V3d>& intersections) const;

	void saveIntersections(const string& filename_1, const string& filename_2) const;

	void saveSDFValue(const string& filename) const;

	void saveCoefficients(const string& filename) const;

	void saveBSplineValue(const string& filename) const;

public:
	void mcVisualization(const string& innerFilename, const V3i& innerResolution,
		const string& outerFilename, const V3i& outerResolution,
		const string& isoFilename, const V3i& isoResolution) const;

	void textureVisualization(const string& filename) const;

	friend class CollisionDetection;
};
