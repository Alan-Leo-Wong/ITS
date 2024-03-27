#pragma once
#include "SVO.h"
#include "BaseModel.h"
#include "ParticleMesh.h"
#include "utils/File.hpp"
#include "test/TestConfig.h"

class ThinShells : public BaseModel, public ParticleMesh
{
	using test_type = Test::type;
private:
	V3d modelOrigin;

	V3i svo_gridSize;
	SparseVoxelOctree svo;
	double voxelWidth;
	vector<V3d> nodeWidthArray;
	//vector<V3d> nodeWidthArray;
	std::map<uint32_t, uint32_t> morton2FineNodeIdx;

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
		svo_gridSize(_grid_x, _grid_y, _grid_z), BaseModel(filename), modelOrigin(modelBoundingBox.boxOrigin), svo(_grid_x, _grid_y, _grid_z)
	{
		svo.createOctree(nModelTris, modelTris, modelBoundingBox, concatFilePath((string)VIS_DIR, modelName));
		treeDepth = svo.treeDepth;
		voxelWidth = svo.svoNodeArray[0].width;
#ifdef IO_SAVE
		saveTree("");
#endif // !IO_SAVE
	}

	ThinShells(const string& filename, const V3i& _grid) :svo_gridSize(_grid), BaseModel(filename), modelOrigin(modelBoundingBox.boxOrigin), svo(_grid)
	{
		svo.createOctree(nModelTris, modelTris, modelBoundingBox, concatFilePath((string)VIS_DIR, modelName));
		treeDepth = svo.treeDepth;
		voxelWidth = svo.svoNodeArray[0].width;
#ifdef IO_SAVE
		saveTree("");
#endif // !IO_SAVE
	}

	ThinShells(const string& filename, const int& _grid_x, const int& _grid_y, const int& _grid_z,
		const bool& _is2UnitCube, const double& _scaleFactor)
		: svo_gridSize(_grid_x, _grid_y, _grid_z), BaseModel(filename, _is2UnitCube, _scaleFactor),
		modelOrigin(modelBoundingBox.boxOrigin), svo(_grid_x, _grid_y, _grid_z)
	{	
		svo.createOctree(nModelTris, modelTris, modelBoundingBox, concatFilePath((string)VIS_DIR, modelName));
		treeDepth = svo.treeDepth;
		voxelWidth = svo.svoNodeArray[0].width;
#ifdef IO_SAVE
		saveTree("");
#endif // !IO_SAVE
	}

	ThinShells(const string& filename, const int& _grid_x, const int& _grid_y, const int& _grid_z,
		const bool& _is2UnitCube, const double& _scaleFactor, const bool& _isAddNoise, const double& noisePercentage)
		: svo_gridSize(_grid_x, _grid_y, _grid_z), BaseModel(filename, _is2UnitCube, _scaleFactor, _isAddNoise, noisePercentage),
		modelOrigin(modelBoundingBox.boxOrigin), svo(_grid_x, _grid_y, _grid_z)
	{
		svo.createOctree(nModelTris, modelTris, modelBoundingBox, concatFilePath((string)VIS_DIR, modelName));
		treeDepth = svo.treeDepth;
		voxelWidth = svo.svoNodeArray[0].width;
#ifdef IO_SAVE
		saveTree("");
#endif // !IO_SAVE
	}

	ThinShells(const string& filename, const V3i& _grid, const bool& _is2UnitCube, const double& _scaleFactor)
		:svo_gridSize(_grid), BaseModel(filename, _is2UnitCube, _scaleFactor), modelOrigin(modelBoundingBox.boxOrigin), svo(_grid)
	{
		svo.createOctree(nModelTris, modelTris, modelBoundingBox, concatFilePath((string)VIS_DIR, modelName));
		treeDepth = svo.treeDepth;
		voxelWidth = svo.svoNodeArray[0].width;
#ifndef IO_SAVE
		saveTree("");
#endif // !IO_SAVE
	}

	~ThinShells() {}

	// ThinShells& operator=(const ThinShells& model);

private:
	V3i getPointDis(const V3d& modelVert, const V3d& origin, const V3d& width) const;

	V3i getPointDis(const V3d& modelVert, const V3d& origin, const double& width) const;

	//void cpIntersectionPoints();
	void cpIntersectionPoints();

	void cpSDFOfTreeNodes();

	void cpCoefficients();

	void cpLatentBSplineValue();

	void initBSplineTree();

	void setLatentMatrix(const double& alpha);

public:
	void creatShell();

	// Octree& bSplineTree() { return bSplineTree; }
	// const Octree& bSplineTree() const { return bSplineTree; }

	std::array<double, 2> getShellIsoVal() { return { innerShellIsoVal, outerShellIsoVal }; }

public:
	void saveTree(const string& filename) const;

	void saveIntersections(const string& filename, const vector<V3d>& intersections) const;

	void saveIntersections(const string& filename_1, const string& filename_2) const;

	void saveSDFValue(const string& filename) const;

	void saveCoefficients(const string& filename) const;

	void saveLatentPoint(const string& filename) const;

	void saveBSplineValue(const string& filename) const;

public:
	void mcVisualization(const string& innerFilename, const V3i& innerResolution,
		const string& outerFilename, const V3i& outerResolution,
		const string& isoFilename, const V3i& isoResolution) const;

	void textureVisualization(const string& filename) const;

	//friend class CollisionDetection;
private:
	vector<std::map<uint32_t, uint32_t>> depthMorton2Nodes;
	vector<std::map<V3d, size_t>> depthVert2Idx;

	void prepareTestDS();

	void prepareMoveOnSurface(int& ac_treeDepth,
		vector<vector<V3d>>& nodeOrigin,
		vector<std::map<uint32_t, size_t>>& morton2Nodes,
		vector<vector<std::array<double, 8>>>& nodeBSplineVal,
		vector<double>& nodeWidth);

	MXd getPointNormal(const MXd& queryPointMat);

	MXd getSurfacePointNormal(const MXd& queryPointMat) override;

	VXd getPointBSplineVal(const MXd& queryPointMat);

	std::pair<VXd, MXd> getPointValGradient(const MXd& before_queryPointMat, const MXd& queryPointMat);

	MXd getProjectPoint(const MXd& before_queryPointMat, const MXd& queryPointMat, const int& iter);

	void lbfgs_optimization(const int& maxIterations, const std::string& out_file) override;

public:
	// ���ڱ���Ĳ�ѯ
	void singlePointQuery(const std::string& out_file, const V3d& point);

	vector<int> multiPointQuery(const vector<V3d>& points, double& time, const int& session, const test_type& choice = Test::CPU);

	void multiPointQuery(const std::string& out_file, const vector<V3d>& points);

	void multiPointQuery(const std::string& out_file, const MXd& points);

	void moveOnSurface(const V3d& modelVert, const V3d& v, const size_t& max_move_cnt);

	void launchParticleSystem(const int& maxIterations, const std::string& out_file);
};
