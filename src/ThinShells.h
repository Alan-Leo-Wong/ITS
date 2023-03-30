#pragma once
#include "Octree.h"
#include "BaseModel.h"
#include <igl/signed_distance.h>

class ThinShells : public BaseModel
{
private:
	Octree bSplineTree;
	int treeDepth;

	double innerShellIsoVal = -DINF;
	double outerShellIsoVal = -DINF;

public:
	//! constructor and destructor
	ThinShells() {};

	ThinShells(const string& filename, const int& _treeDepth) : BaseModel(filename), treeDepth(_treeDepth)
	{
		bSplineTree = Octree(_treeDepth);
		bSplineTree.createOctree(modelBoundingBox);
	}

	~ThinShells() {};

	ThinShells& operator=(const ThinShells& model);

private:
	void cpIntersectionPoints();

	void cpSDFOfTreeNodes();

	void cpCoefficients();

	void cpBSplineValue();

	void initBSplineTree();

public:
	void creatShell();

	void mcVisualization(const string& innerFilename, const V3i& innerResolution,
						 const string& outerFilename, const V3i& outerResolution) const;

	void textureVisualization(const string& filename) const;

	//Octree& bSplineTree() { return bSplineTree; }
	//const Octree& bSplineTree() const { return bSplineTree; }

	std::array<double, 2> getShellIsoVal() { return { innerShellIsoVal, outerShellIsoVal }; }

public:
	//! I/O
	void saveInterPoints(string& filename_1, string& filename_2);

	void SaveBValue2TXT(string& filename, VXd X);

	friend class CollisionDetection;
};

