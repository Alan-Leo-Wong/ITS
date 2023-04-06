#pragma once
#include "Octree.h"
#include "BaseModel.h"
#include "SDFHelper.h"

class ThinShells : public BaseModel
{
private:
	Octree bSplineTree;
	int treeDepth;

	vector<V3d> edgeInterPoints; // Intersection points of octree node and mesh's edges
	vector<V3d> faceInterPoints; // Intersection points of octree node's edges and mesh's faces
	vector<V3d> allInterPoints;  // All intersection points of octree node and mesh
	vector<OctreeNode*> interLeafNodes;

private:
	fcpw::Scene<3> scene;
	VXd sdfVal;
	VXd lambda;
	VXd bSplineVal;
private:
	double innerShellIsoVal = -DINF;
	double outerShellIsoVal = -DINF;

public:
	// constructor and destructor
	ThinShells() {}

	ThinShells(const string& filename, const int& _treeDepth) : BaseModel(filename), treeDepth(_treeDepth),
		bSplineTree(_treeDepth, modelBoundingBox, nModelVerts, modelVerts)
	{
		initSDF(scene, modelVerts, modelFaces);
		refineSurfaceTree();
		//bSplineTree = Octree(_treeDepth, modelBoundingBox, nModelVerts, modelVerts);
		//cout << bSplineTree.allNodes[0]->depth << endl;
		saveOctree("");
	}

	~ThinShells() {}

	// ThinShells& operator=(const ThinShells& model);

private:
	void refineSurfaceTree();

	void cpIntersectionPoints();

	void cpSDFOfTreeNodes();

	void cpCoefficients();

	void cpBSplineValue();

	void initBSplineTree();

public:
	void creatShell();

	// Octree& bSplineTree() { return bSplineTree; }
	// const Octree& bSplineTree() const { return bSplineTree; }

	std::array<double, 2> getShellIsoVal() { return { innerShellIsoVal, outerShellIsoVal }; }

public:
	void saveOctree(const string& filename) const;

	void saveIntersections(const string& filename, const vector<V3d>& intersections) const;

	void saveIntersections(const string& filename_1, const string& filename_2) const;

	void saveSDFValue(const string& filename) const;

	void saveCoefficients(const string& filename) const;

	void saveBSplineValue(const string& filename) const;

public:
	void mcVisualization(const string& innerFilename, const V3i& innerResolution,
		const string& outerFilename, const V3i& outerResolution) const;

	void textureVisualization(const string& filename) const;

	friend class CollisionDetection;
};
