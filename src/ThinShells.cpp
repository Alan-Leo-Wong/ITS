#include "ThinShells.h"
#include "BSpline.hpp"
#include "utils\Common.hpp"
#include "utils\String.hpp"
#include "utils\cuda\CUDAMath.hpp"
#include "cuAcc\MarchingCubes\MarchingCubes.h"
#include <queue>
#include <iomanip>
#include <Eigen\Sparse>
#include <igl\signed_distance.h>

//////////////////////
//  Create  Shells  //
//////////////////////
inline void ThinShells::cpIntersectionPoints()
{
	vector<V2i> modelEdges = extractEdges();
	uint nModelEdges = modelEdges.size();

	const size_t& numFineNodes = svo.numFineNodes;
	cout << "-- Number of level-0 nodes: " << numFineNodes << endl;;

	const vector<SVONode>& nodeArray = svo.svoNodeArray;
	const vector<node_edge_type>& fineNodeEdges = svo.fineNodeEdgeArray;

	// 只需要求三角形与最底层节点的交点
	
	// 三角形的边与node面交， 因为隐式B样条基定义在了left/bottom/back corner上， 所以与节点只需要求与这三个面的交即可
	std::cout << "1. Computing the intersections between mesh EDGES and nodes...\n";
	for (int i = 0; i < nModelEdges; i++)
	{
		Eigen::Vector2i e = modelEdges[i];
		V3d p1 = m_V.row(e.x()); V3d p2 = m_V.row(e.y());
		V3d modelEdgeDir = p2 - p1;

		for (int j = 0; j < numFineNodes; ++j)
		{
			V3d lbbCorner = nodeArray[j].origin;
			double width = nodeArray[j].width;

			// back plane
			double back_t = DINF;
			if (modelEdgeDir.x() != 0)
				back_t = (lbbCorner.x() - p1.x()) / modelEdgeDir.x();
			// left plane
			double left_t = DINF;
			if (modelEdgeDir.y() != 0)
				left_t = (lbbCorner.y() - p1.y()) / modelEdgeDir.y();
			// bottom plane
			double bottom_t = DINF;
			if (modelEdgeDir.z() != 0)
				bottom_t = (lbbCorner.z() - p1.z()) / modelEdgeDir.z();

			if (isInRange(.0, 1.0, back_t) &&
				isInRange(lbbCorner.y(), lbbCorner.y() + width, (p1 + back_t * modelEdgeDir).y()) &&
				isInRange(lbbCorner.z(), lbbCorner.z() + width, (p1 + back_t * modelEdgeDir).z()))
			{
				edgeInterPoints.emplace_back(p1 + back_t * modelEdgeDir);
			}
			if (isInRange(.0, 1.0, left_t) &&
				isInRange(lbbCorner.x(), lbbCorner.x() + width, (p1 + left_t * modelEdgeDir).x()) &&
				isInRange(lbbCorner.z(), lbbCorner.z() + width, (p1 + left_t * modelEdgeDir).z()))
			{
				edgeInterPoints.emplace_back(p1 + left_t * modelEdgeDir);
			}
			if (isInRange(.0, 1.0, bottom_t) &&
				isInRange(lbbCorner.x(), lbbCorner.x() + width, (p1 + bottom_t * modelEdgeDir).x()) &&
				isInRange(lbbCorner.y(), lbbCorner.y() + width, (p1 + bottom_t * modelEdgeDir).y()))
			{
				edgeInterPoints.emplace_back(p1 + bottom_t * modelEdgeDir);
			}
		}
	}

	cout << "-- 三角形边与node的交点数量：" << edgeInterPoints.size() << endl;
	allInterPoints.insert(allInterPoints.end(), edgeInterPoints.begin(), edgeInterPoints.end());

	// 三角形面与node边线交（有重合点）
	std::cout << "2. Computing the intersections between mesh FACES and node EDGES..." << endl;
	for (const auto& tri : modelTris)
	{
		V3d triEdge_1 = tri.p2 - tri.p1; V3d triEdge_2 = tri.p3 - tri.p2; V3d triEdge_3 = tri.p1 - tri.p3;
		V3d triNormal = tri.normal; double triDir = tri.dir;
		for (const auto& nodeEdge : fineNodeEdges)
		{
			thrust_edge_type edge = nodeEdge.first;
			V3d edgeDir = edge.second - edge.first;

			if (fabsf(triNormal.dot(edgeDir)) < 1e-9) continue;

			double t = (-triDir - triNormal.dot(edge.first)) / (triNormal.dot(edgeDir));
			if (t <= 0. || t >= 1.) continue;
			V3d interPoint = edge.first + edgeDir * t;

			if (triEdge_1.cross(interPoint - tri.p1).dot(triNormal) < 0) continue;
			if (triEdge_2.cross(interPoint - tri.p2).dot(triNormal) < 0) continue;
			if (triEdge_3.cross(interPoint - tri.p3).dot(triNormal) < 0) continue;

			faceInterPoints.emplace_back(interPoint);
		}
	}

	//faceInterPoints.erase(std::unique(faceInterPoints.begin(), faceInterPoints.end()), faceInterPoints.end());
	cout << "-- 三角形面与node边的交点数量：" << faceInterPoints.size() << endl;

	allInterPoints.insert(allInterPoints.end(), faceInterPoints.begin(), faceInterPoints.end());

	allInterPoints.erase(std::unique(allInterPoints.begin(), allInterPoints.end()), allInterPoints.end());
	cout << "-- 总交点数量：" << allInterPoints.size() << endl;
}

inline void ThinShells::cpSDFOfTreeNodes()
{
	const uint nAllNodes = bSplineTree.nAllNodes;

	// initialize a 3d scene
	//sdfVal.resize(nAllNodes * 8);
	sdfVal.resize(nAllNodes * 8 + allInterPoints.size());
	sdfVal.setZero();
	//sdfVal.resize(nLeafNodes * 8);

	for (int i = 0; i < nAllNodes; ++i)
		//for (int i = 0; i < nLeafNodes; ++i)
		for (int k = 0; k < 8; ++k)
			sdfVal(i * 8 + k) = getSignedDistance(bSplineTree.allNodes[i]->corners[k], scene);
}

inline void ThinShells::cpCoefficients()
{
	using SpMat = Eigen::SparseMatrix<double>;
	using Trip = Eigen::Triplet<double>;

	const auto allNodes = bSplineTree.allNodes;
	const uint nAllNodes = bSplineTree.nAllNodes;
	auto corner2IDs = bSplineTree.corner2IDs;

	// initial matrix
	//SpMat sm(nAllNodes * 8, nAllNodes * 8); // A
	SpMat sm(nAllNodes * 8 + allInterPoints.size(), nAllNodes * 8); // A
	vector<Trip> matVal;

	for (int i = 0; i < nAllNodes; ++i)
	{
		auto node_i = allNodes[i];
		//auto node_i = leafNodes[i];
		map<size_t, bool> visID;
		auto [inDmPoints, inDmPointsID] = bSplineTree.setInDomainPoints(node_i, visID);
		const int nInDmPoints = inDmPoints.size();

		for (int k = 0; k < 8; ++k)
		{
			V3d i_corner = node_i->corners[k];
			const uint ic_row = i * 8 + k;

			for (const auto& id_ck : corner2IDs[i_corner])
			{
				// i_corner所在的其他节点的id和位置
				const uint o_id = id_ck.first;
				const uint o_k = id_ck.second;
				const uint o_realID = o_id * 8 + o_k;

				if (!visID[o_realID]) matVal.emplace_back(Trip(ic_row, o_realID, 1));
			}

			for (int j = 0; j < nInDmPoints; ++j)
			{
				double val = BaseFunction4Point(inDmPoints[j].first, inDmPoints[j].second, i_corner);
				assert(inDmPointsID[j] < nAllNodes * 8, "index of col > nAllNodes * 8!");
				if (val != 0) matVal.emplace_back(Trip(ic_row, inDmPointsID[j], val));
			}
		}
	}

	for (int i = 0; i < allInterPoints.size(); ++i)
	{
		for (int j = 0; j < nAllNodes; ++j)
		{
			for (int k = 0; k < 8; ++k)
			{
				double val = BaseFunction4Point(allNodes[j]->corners[k], allNodes[j]->width, allInterPoints[i]);
				matVal.emplace_back(Trip(nAllNodes * 8 + i, j * 8 + k, val));
			}
		}
	}

	sm.setFromTriplets(matVal.begin(), matVal.end());
	//sm.makeCompressed();

	auto A = sm;
	auto b = sdfVal;
	/*auto A = sm.transpose() * sm;
	auto b = sm.transpose() * sdfVal;*/

	Eigen::LeastSquaresConjugateGradient<SpMat> lscg;
	lscg.compute(A);
	lambda = lscg.solve(b);

	cout << "-- Residual Error: " << (A * lambda - b).norm() << endl;

	//saveCoefficients(concatFilePath((string)OUT_DIR, modelName, std::to_string(maxDepth), (string)"Coefficients.txt"));
}

inline void ThinShells::cpBSplineValue()
{
	const uint nAllNodes = bSplineTree.nAllNodes;

	const uint nInterPoints = allInterPoints.size();
	const uint nInterLeafNodes = interLeafNodes.size();

	bSplineVal.resize(nModelVerts + nInterPoints);
	bSplineVal.setZero();

	for (int i = 0; i < nModelVerts; ++i)
	{
		V3d modelPoint = modelVerts[i];

		for (int j = 0; j < nAllNodes; ++j)
		{
			auto node = bSplineTree.allNodes[j];
			for (int k = 0; k < 8; k++)
				bSplineVal[i] += lambda[j * 8 + k] * (BaseFunction4Point(node->corners[k], node->width, modelPoint));
		}
	}

	int cnt = 0;
	for (int i = 0; i < nInterPoints; ++i)
	{
		cnt = i + nModelVerts;
		V3d interPoint = allInterPoints[i];

		for (int j = 0; j < nAllNodes; ++j)
		{
			auto node = bSplineTree.allNodes[j];
			for (int k = 0; k < 8; k++)
				bSplineVal[cnt] += lambda[j * 8 + k] * (BaseFunction4Point(node->corners[k], node->width, interPoint));
		}
	}

	innerShellIsoVal = *(std::min_element(bSplineVal.begin(), bSplineVal.end()));
	outerShellIsoVal = *(std::max_element(bSplineVal.begin(), bSplineVal.end()));

	//bSplineTree.saveBSplineValue(concatFilePath((string)OUT_DIR, modelName, std::to_string(treeDepth), (string)"bSplineVal.txt"));
}

inline void ThinShells::initBSplineTree()
{
	cout << "\nComputing intersection points of " << std::quoted(modelName) << "and leaf nodes...\n=====================" << endl;
	cpIntersectionPoints();
	cout << "=====================\n";
	saveIntersections("", "");

	cout << "\nComputing discrete SDF of tree nodes..." << endl;
	cpSDFOfTreeNodes();
	cout << "=====================\n";
	saveSDFValue("");

	cout << "\nComputing coefficients..." << endl;
	cpCoefficients();
	cout << "=====================\n";
	saveCoefficients("");

	cout << "\nComputing B-Spline value..." << endl;
	cpBSplineValue();
	cout << "=====================\n";
	saveBSplineValue("");
}

void ThinShells::creatShell()
{
	initBSplineTree();
}

//////////////////////
//  I/O: Save Data  //
//////////////////////
void ThinShells::saveOctree(const string& filename) const
{
	string t_filename = filename;
	if (filename.empty())
		t_filename = concatFilePath((string)VIS_DIR, modelName, std::to_string(treeDepth), (string)"octree.obj");
	bSplineTree.saveNodeCorners2OBJFile(t_filename);
}

void ThinShells::saveIntersections(const string& filename, const vector<V3d>& intersections) const
{
	std::ofstream out(filename);
	if (!out) { fprintf(stderr, "IO Error: File %s could not be opened!", filename.c_str()); return; }
	checkDir(filename);

	for (const V3d& p : intersections)
		out << p.x() << " " << p.y() << " " << p.z() << endl;
	out.close();
}

void ThinShells::saveIntersections(const string& filename_1, const string& filename_2) const
{
	string t_filename = filename_1;
	if (filename_1.empty())
		t_filename = concatFilePath((string)VIS_DIR, modelName, std::to_string(treeDepth), (string)"edgeInter.xyz");
	cout << "-- Save mesh EDGES and octree Nodes to " << std::quoted(t_filename) << endl;
	saveIntersections(t_filename, edgeInterPoints);

	t_filename = filename_2;
	if (filename_2.empty())
		t_filename = concatFilePath((string)VIS_DIR, modelName, std::to_string(treeDepth), (string)"faceInter.xyz");
	cout << "-- Save mesh FACES and octree node EDGES to " << std::quoted(t_filename) << endl;
	saveIntersections(t_filename, faceInterPoints);
}

void ThinShells::saveSDFValue(const string& filename) const
{
	string t_filename = filename;
	if (filename.empty())
		t_filename = concatFilePath((string)OUT_DIR, modelName, std::to_string(treeDepth), (string)"SDFValue.txt");

	checkDir(t_filename);
	std::ofstream out(t_filename);
	if (!out) { fprintf(stderr, "IO Error: File %s could not open!", filename.c_str()); return; }

	cout << "-- Save SDF value to " << std::quoted(t_filename) << endl;
	for (const auto& val : sdfVal)
		out << val << endl;
	out.close();
}

void ThinShells::saveCoefficients(const string& filename) const
{
	string t_filename = filename;
	if (filename.empty())
		t_filename = concatFilePath((string)OUT_DIR, modelName, std::to_string(treeDepth), (string)"Coefficients.txt");

	checkDir(t_filename);
	std::ofstream out(t_filename);
	if (!out) { fprintf(stderr, "IO Error: File %s could not open!", t_filename.c_str()); return; }

	cout << "-- Save coefficients to " << std::quoted(t_filename) << endl;
	for (const auto& val : lambda)
		out << val << endl;
}

void ThinShells::saveBSplineValue(const string& filename) const
{
	string t_filename = filename;
	if (filename.empty())
		t_filename = concatFilePath((string)OUT_DIR, modelName, std::to_string(treeDepth), (string)"BSplineValue.txt");

	checkDir(t_filename);
	std::ofstream out(t_filename);
	if (!out) { fprintf(stderr, "IO Error: File %s could not open!", t_filename.c_str()); return; }

	cout << "-- Save B-Spline value to " << std::quoted(t_filename) << endl;
	out << std::setiosflags(std::ios::fixed) << std::setprecision(9) << bSplineVal << endl;
	out.close();
}

//////////////////////
//   Visualiztion   //
//////////////////////
void ThinShells::mcVisualization(const string& innerFilename, const V3i& innerResolution,
	const string& outerFilename, const V3i& outerResolution) const
{
	V3d gridOrigin = modelBoundingBox.boxOrigin;
	V3d gridWidth = modelBoundingBox.boxWidth;

	if (!innerFilename.empty() && innerShellIsoVal != -DINF)
	{
		cout << "\nExtract inner shell by MarchingCubes..." << endl;
		MC::marching_cubes(bSplineTree.allNodes, lambda, make_double3(gridOrigin), make_double3(gridWidth),
			make_uint3(innerResolution), innerShellIsoVal, innerFilename);
		cout << "=====================\n";
	}

	if (!outerFilename.empty() && outerShellIsoVal != -DINF)
	{
		cout << "\nExtract outer shell by MarchingCubes..." << endl;
		MC::marching_cubes(bSplineTree.allNodes, lambda, make_double3(gridOrigin), make_double3(gridWidth),
			make_uint3(outerResolution), outerShellIsoVal, outerFilename);
		cout << "=====================\n";
	}
}

void ThinShells::textureVisualization(const string& filename) const
{
	writeTexturedObjFile(filename, bSplineVal);
}