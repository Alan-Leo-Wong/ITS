#include "ThinShells.h"
#include "BSpline.hpp"
#include "utils\Common.hpp"
#include "utils\String.hpp"
#include "utils\cuda\CUDAMath.hpp"
#include "cuAcc\MarchingCubes\MarchingCubes.h"
#include <queue>
#include <iomanip>
#include <numeric>
#include <Eigen\Sparse>
#include <igl\signed_distance.h>

//////////////////////
//  Create  Shells  //
//////////////////////
// idx: subdepth
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
			if (t < 0. || t > 1.) continue;
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
	const auto& nodeVertexArray = svo.nodeVertexArray;
	const size_t& numNodeVerts = svo.numNodeVerts;
	MXd pointsMat(numNodeVerts, 3);
	for (int i = 0; i < numNodeVerts; ++i) pointsMat.row(i) = nodeVertexArray[i].first;

	VXd S;
	{
		VXi I;
		MXd C, N;
		igl::signed_distance(pointsMat, m_V, m_F, igl::SignedDistanceType::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER, S, I, C, N);
		// Convert distances to binary inside-outside data --> aliasing artifacts
		sdfVal = S;
		std::for_each(sdfVal.data(), sdfVal.data() + sdfVal.size(), [](double& b) {b = (b > 0 ? 1 : (b < 0 ? -1 : 0)); });
	}
}

inline void ThinShells::cpCoefficients()
{
	using SpMat = Eigen::SparseMatrix<double>;
	using Trip = Eigen::Triplet<double>;

	const vector<vector<node_vertex_type>>& depthNodeVertexArray = svo.depthNodeVertexArray;
	const vector<size_t>& esumDepthNodeVerts = svo.esumDepthNodeVerts;
	vector<std::map<V3d, size_t>> nodeVertex2Idx(treeDepth);

	for (int d = 0; d < treeDepth; ++d)
	{
		const size_t d_numNodeVerts = depthNodeVertexArray[d].size();
		vector<size_t> d_nodeVertexIdx(d_numNodeVerts);
		std::iota(d_nodeVertexIdx.begin(), d_nodeVertexIdx.end(), d_numNodeVerts);

		std::transform(depthNodeVertexArray[d].begin(), depthNodeVertexArray[d].end(), d_nodeVertexIdx.begin(),
			std::inserter(nodeVertex2Idx[d], nodeVertex2Idx[d].end()),
			[](const node_vertex_type& val, size_t i) {
				return std::make_pair(val.first, i);
			});

		std::cout << "depth = " << d << ":" << std::endl;
		for (const auto& entry : nodeVertex2Idx[d])
		{
			std::cout << "{" << entry.first << ", " << entry.second << "}" << std::endl;
		}
		std::cout << "----------" << std::endl;
	}

	// initial matrix
	const size_t& numNodeVerts = svo.numNodeVerts;
	SpMat sm(numNodeVerts, numNodeVerts); // A
	//SpMat sm(nAllNodes * 8 + allInterPoints.size(), nAllNodes * 8); // A
	vector<Trip> matVal;

	for (int d = 0; d < treeDepth; ++d)
	{
		const size_t d_numNodeVerts = depthNodeVertexArray[d].size();
		const size_t& d_esumNodeVerts = esumDepthNodeVerts[d];
		for (int i = 0; i < d_numNodeVerts; ++i)
		{
			V3d i_nodeVertex = depthNodeVertexArray[d][i].first;
			uint32_t i_fromNodeIdx = depthNodeVertexArray[d][i].second;

			for (int j = d; j >= 0; --j)
				matVal.emplace_back(Trip(esumDepthNodeVerts[j] + nodeVertex2Idx[j][i_nodeVertex], esumDepthNodeVerts[j] + nodeVertex2Idx[j][i_nodeVertex], 1)); // self and child

			// parent
			auto [inDmPoints, inDmPointsIdx] = svo.setInDomainPoints(i_fromNodeIdx, d, esumDepthNodeVerts, nodeVertex2Idx);
			const int nInDmPoints = inDmPoints.size();

			for (int k = 0; k < nInDmPoints; ++k)
			{
				double val = BaseFunction4Point(inDmPoints[k].first, inDmPoints[k].second, i_nodeVertex);
				assert(inDmPointsIdx[k] < numNodeVerts, "index of col > numNodeVertex!");
				if (val != 0) matVal.emplace_back(Trip(i, inDmPointsIdx[k], val));
			}
		}
	}

	/*for (int i = 0; i < allInterPoints.size(); ++i)
	{
		for (int j = 0; j < nAllNodes; ++j)
		{
			for (int k = 0; k < 8; ++k)
			{
				double val = BaseFunction4Point(allNodes[j]->corners[k], allNodes[j]->width, allInterPoints[i]);
				matVal.emplace_back(Trip(nAllNodes * 8 + i, j * 8 + k, val));
			}
		}
	}*/

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
	const uint nInterPoints = allInterPoints.size();

	bSplineVal.resize(nModelVerts + nInterPoints);
	bSplineVal.setZero();

	const vector<SVONode>& svoNodeArray = svo.svoNodeArray;
	const vector<vector<node_vertex_type>>& depthNodeVertexArray = svo.depthNodeVertexArray;
	const vector<size_t>& esumDepthNodeVerts = svo.esumDepthNodeVerts;

	for (int i = 0; i < nModelVerts; ++i)
	{
		const V3d& modelVert = modelVerts[i];
		for (int d = 0; d < treeDepth; ++d)
		{
			const size_t d_numNodeVerts = depthNodeVertexArray[d].size();
			const size_t& d_esumNodeVerts = esumDepthNodeVerts[d];
			for (int j = 0; j < d_numNodeVerts; ++j)
			{
				V3d nodeVert = depthNodeVertexArray[d][j].first;
				uint32_t nodeIdx = depthNodeVertexArray[d][j].second;
				bSplineVal[i] += lambda[d_esumNodeVerts + j] * (BaseFunction4Point(nodeVert, svoNodeArray[nodeIdx].width, modelVert));
			}
		}
	}

	int cnt = 0;
	for (int i = 0; i < nInterPoints; ++i)
	{
		cnt = i + nModelVerts;
		const V3d& interPoint = allInterPoints[i];
		for (int d = 0; d < treeDepth; ++d)
		{
			const size_t d_numNodeVerts = depthNodeVertexArray[d].size();
			const size_t& d_esumNodeVerts = esumDepthNodeVerts[d];
			for (int j = 0; j < d_numNodeVerts; ++j)
			{
				V3d nodeVert = depthNodeVertexArray[d][j].first;
				uint32_t nodeIdx = depthNodeVertexArray[d][j].second;
				bSplineVal[cnt] += lambda[d_esumNodeVerts + j] * (BaseFunction4Point(nodeVert, svoNodeArray[nodeIdx].width, interPoint));
			}
		}
	}

	innerShellIsoVal = *(std::min_element(bSplineVal.begin(), bSplineVal.end()));
	outerShellIsoVal = *(std::max_element(bSplineVal.begin(), bSplineVal.end()));

	//bSplineTree.saveBSplineValue(concatFilePath((string)OUT_DIR, modelName, std::to_string(treeDepth), (string)"bSplineVal.txt"));
}

inline void ThinShells::initBSplineTree()
{
	cout << "\nComputing intersection points of " << std::quoted(modelName) << "and level-0 nodes...\n=====================" << endl;
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
void ThinShells::saveTree(const string& filename) const
{
	string t_filename = filename;
	if (filename.empty()) t_filename = concatFilePath((string)VIS_DIR, modelName, std::to_string(treeDepth), (string)"_svo.obj");

	svo.saveSVO(t_filename);
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
		cout << "\n[MC] Extract inner shell by MarchingCubes..." << endl;
		MC::marching_cubes(svo.depthNodeVertexArray, svo.svoNodeArray, svo.esumDepthNodeVerts, svo.numNodeVerts, lambda, make_double3(gridOrigin), make_double3(gridWidth),
			make_uint3(innerResolution), innerShellIsoVal, innerFilename);
		cout << "=====================\n";
	}

	if (!outerFilename.empty() && outerShellIsoVal != -DINF)
	{
		cout << "\n[MC] Extract outer shell by MarchingCubes..." << endl;
		MC::marching_cubes(svo.depthNodeVertexArray, svo.svoNodeArray, svo.esumDepthNodeVerts, svo.numNodeVerts, lambda, make_double3(gridOrigin), make_double3(gridWidth),
			make_uint3(outerResolution), outerShellIsoVal, outerFilename);
		cout << "=====================\n";
	}
}

void ThinShells::textureVisualization(const string& filename) const
{
	writeTexturedObjFile(filename, bSplineVal);
}