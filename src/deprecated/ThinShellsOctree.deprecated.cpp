#include "ThinShells.h"
#include "BSpline.hpp"
#include "utils\Common.hpp"
#include "utils\File.hpp"
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

	// ֻ��Ҫ������������ײ�ڵ�Ľ���
	
	// �����εı���node�潻�� ��Ϊ��ʽB��������������left/bottom/back corner�ϣ� ������ڵ�ֻ��Ҫ������������Ľ�����
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

	cout << "-- �����α���node�Ľ���������" << edgeInterPoints.size() << endl;
	allInterPoints.insert(allInterPoints.end(), edgeInterPoints.begin(), edgeInterPoints.end());

	// ����������node���߽������غϵ㣩
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
	cout << "-- ����������node�ߵĽ���������" << faceInterPoints.size() << endl;

	allInterPoints.insert(allInterPoints.end(), faceInterPoints.begin(), faceInterPoints.end());

	allInterPoints.erase(std::unique(allInterPoints.begin(), allInterPoints.end()), allInterPoints.end());
	cout << "-- �ܽ���������" << allInterPoints.size() << endl;
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
				// i_corner���ڵ������ڵ��id��λ��
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

///Deprecated: cpIntersectionPoints
//inline void ThinShells::cpIntersectionPoints()
//{
//	vector<V2i> modelEdges = extractEdges();
//	uint nModelEdges = modelEdges.size();
//	cout << "-- Number of leaf nodes: " << bSplineTree.nLeafNodes << endl;;
//
//	auto leafNodes = bSplineTree.leafNodes;
//
//	// �����εı���node�潻�� ��Ϊ��ʽB��������������left/bottom/back corner�ϣ� ������ڵ�ֻ��Ҫ������������Ľ�����
//	std::cout << "1. Computing the intersections between mesh EDGES and nodes...\n";
//	for (int i = 0; i < nModelEdges; i++)
//	{
//		Eigen::Vector2i e = modelEdges[i];
//		V3d p1 = m_V.row(e.x());
//		V3d p2 = m_V.row(e.y());
//		V3d dir = p2 - p1;
//
//		for (auto node : leafNodes)
//		{
//			V3d lbbCorner = node->boundary.first;
//			V3d width = node->width;
//
//			// back plane
//			double back_t = DINF;
//			if (dir.x() != 0)
//				back_t = (lbbCorner.x() - p1.x()) / dir.x();
//			// left plane
//			double left_t = DINF;
//			if (dir.y() != 0)
//				left_t = (lbbCorner.y() - p1.y()) / dir.y();
//			// bottom plane
//			double bottom_t = DINF;
//			if (dir.z() != 0)
//				bottom_t = (lbbCorner.z() - p1.z()) / dir.z();
//
//			if (isInRange(.0, 1.0, back_t) &&
//				isInRange(lbbCorner.y(), lbbCorner.y() + width.y(), (p1 + back_t * dir).y()) &&
//				isInRange(lbbCorner.z(), lbbCorner.z() + width.z(), (p1 + back_t * dir).z()))
//			{
//				edgeInterPoints.emplace_back(p1 + back_t * dir);
//				node->isInterMesh = true;
//			}
//			if (isInRange(.0, 1.0, left_t) &&
//				isInRange(lbbCorner.x(), lbbCorner.x() + width.x(), (p1 + left_t * dir).x()) &&
//				isInRange(lbbCorner.z(), lbbCorner.z() + width.z(), (p1 + left_t * dir).z()))
//			{
//				edgeInterPoints.emplace_back(p1 + left_t * dir);
//				node->isInterMesh = true;
//			}
//			if (isInRange(.0, 1.0, bottom_t) &&
//				isInRange(lbbCorner.x(), lbbCorner.x() + width.x(), (p1 + bottom_t * dir).x()) &&
//				isInRange(lbbCorner.y(), lbbCorner.y() + width.y(), (p1 + bottom_t * dir).y()))
//			{
//				edgeInterPoints.emplace_back(p1 + bottom_t * dir);
//				node->isInterMesh = true;
//			}
//		}
//	}
//
//	cout << "-- �����α���node�Ľ���������" << edgeInterPoints.size() << endl;
//
//	allInterPoints.insert(allInterPoints.end(), edgeInterPoints.begin(), edgeInterPoints.end());
//
//	// ����������node���߽������غϵ㣩
//	std::cout << "2. Computing the intersections between mesh FACES and node EDGES..." << endl;
//	for (const auto& leafNode : leafNodes)
//	{
//		auto edges = leafNode->edges;
//
//		for (int i = 0; i < nModelFaces; i++)
//		{
//			V3i f = modelFaces[i];
//			V3d p1 = modelVerts[f.x()];
//			V3d p2 = modelVerts[f.y()];
//			V3d p3 = modelVerts[f.z()];
//			Eigen::Matrix3d faceMatrix;
//			faceMatrix << p1.x(), p2.x(), p3.x(),
//				p1.y(), p2.y(), p3.y(),
//				p1.z(), p2.z(), p3.z();
//			V3d maxElement = faceMatrix.rowwise().maxCoeff();
//			V3d minElement = faceMatrix.rowwise().minCoeff();
//
//			Triangle t(p1, p2, p3);
//
//			for (int j = 0; j < 4; ++j)
//			{
//				auto edge = edges[j];
//				double y = edge.first.y();
//				double z = edge.first.z();
//				if (maxElement.x() <= edge.first.x() || minElement.x() >= edge.second.x()) continue;
//				if (minElement.y() >= y || maxElement.y() <= y || minElement.z() >= z || maxElement.z() <= z) continue;
//				V3d coefficient = t.cpCoefficientOfTriangle(y, z, 1);
//				if (coefficient.x() > 0)
//				{
//					double interX = coefficient.x() * p1.x() + coefficient.y() * p2.x() + coefficient.z() * p3.x();
//					if (interX >= edge.first.x() && interX <= edge.second.x())
//					{
//						faceInterPoints.emplace_back(V3d(interX, y, z));
//						leafNode->isInterMesh = true;
//					}
//				}
//			}
//
//			for (int j = 4; j < 8; ++j)
//			{
//				auto edge = edges[j];
//				double x = edge.first.x();
//				double z = edge.first.z();
//				if (maxElement.y() <= edge.first.y() || minElement.y() >= edge.second.y()) continue;
//				if (minElement.x() >= x || maxElement.x() <= x || minElement.z() >= z || maxElement.z() <= z) continue;
//				V3d coefficient = t.cpCoefficientOfTriangle(z, x, 2);
//				if (coefficient.x() > 0)
//				{
//					double interY = coefficient.x() * p1.y() + coefficient.y() * p2.y() + coefficient.z() * p3.y();
//					if (interY >= edge.first.y() && interY <= edge.second.y())
//					{
//						faceInterPoints.emplace_back(V3d(x, interY, z));
//						leafNode->isInterMesh = true;
//					}
//				}
//			}
//
//			for (int j = 8; j < 11; ++j)
//			{
//				auto edge = edges[j];
//				double x = edge.first.x();
//				double y = edge.first.y();
//				if (maxElement.z() <= edge.first.z() || minElement.z() >= edge.second.z()) continue;
//				if (minElement.x() >= x || maxElement.x() <= x || minElement.y() >= y || maxElement.y() <= y) continue;
//				V3d coefficient = t.cpCoefficientOfTriangle(x, y, 0);
//				if (coefficient.x() > 0)
//				{
//					double interZ = coefficient.x() * p1.z() + coefficient.y() * p2.z() + coefficient.z() * p3.z();
//					if (interZ >= edge.first.z() && interZ <= edge.second.z())
//					{
//						faceInterPoints.emplace_back(V3d(x, y, interZ));
//						leafNode->isInterMesh = true;
//					}
//				}
//			}
//		}
//
//		if (leafNode->isInterMesh) interLeafNodes.emplace_back(leafNode); // ɸѡ�н����Ҷ�ӽڵ�
//	}
//
//	faceInterPoints.erase(std::unique(faceInterPoints.begin(), faceInterPoints.end()), faceInterPoints.end());
//	cout << "-- ����������node�ߵĽ���������" << faceInterPoints.size() << endl;
//
//	allInterPoints.insert(allInterPoints.end(), faceInterPoints.begin(), faceInterPoints.end());
//
//	allInterPoints.erase(std::unique(allInterPoints.begin(), allInterPoints.end()), allInterPoints.end());
//	cout << "-- �ܽ���������" << allInterPoints.size() << endl;
//}

///Deprecated: refineSurface
//// idx: subdepth
//constexpr size_t EDGE_VAL_LR[] = { 1, 7, 55, 439, 3511, 28087, 224695, 1797559, 14380471 };
//constexpr size_t EDGE_VAL_BF[] = { 2, 14, 110, 878, 7022, 56174, 449390, 3595118, 28760942 };
//constexpr size_t EDGE_VAL_BU[] = { 4, 28, 220, 1756, 14044, 112348, 898780, 7190236, 57521884 };
//void ThinShells::refineSurfaceTree()
//{
//	if (bSplineTree.d_leafNodes.empty() || treeDepth <= 2) return;
//
//	const V3d d_leafNodeWidth = bSplineTree.d_leafNodes[0]->width;
//	const V3d treeOrigin = bSplineTree.treeOrigin;
//	auto& nAllNodes = bSplineTree.nAllNodes;
//	auto& allNodes = bSplineTree.allNodes;
//	auto& nLeafNodes = bSplineTree.nLeafNodes;
//	auto& leafNodes = bSplineTree.leafNodes;
//	auto& d_leafNodes = bSplineTree.d_leafNodes;
//	auto& id2Node = bSplineTree.id2Node;
//	auto& visNodeId = bSplineTree.visNodeId;
//
//	// �ж��Ƿ񴩹�����
//	/*
//	* @param width: d_leafNodes���
//	*/
//	auto isNodeCrossSurface = [=](OctreeNode* node)->bool
//	{
//		//// �õ�nodeId��Ӧ��ԭ��λ��
//		//MXd corners(8, 3);
//		//for (int i = 0; i < 8; ++i)
//		//	corners.row(i) = node->corners[i];
//		//VXd S, B;
//		//{
//		//	VXi I;
//		//	MXd C, N;
//		//	igl::signed_distance(corners, m_V, m_F, igl::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER, S, I, C, N);
//		//}
//		//node->sdf[0] = S(0);
//		//bool ref = (node->sdf[0]) > 0;
//		//bool flag = false;
//		//for (int i = 1; i < 8; i++)
//		//{
//		//	node->sdf[i] = S(i);
//		//	if (!flag && ((node->sdf[i] < 0) == ref)) flag = true;
//		//}
//
//		V3d originCorner = node->corners[0];
//		node->sdf[0] = getSignedDistance(originCorner, scene);
//
//		bool ref = (node->sdf[0]) > 0;
//		bool flag = false;
//		for (int i = 1; i < 8; i++)
//		{
//			node->sdf[i] = getSignedDistance(node->corners[i], scene);
//			if (!flag && ((node->sdf[i] < 0) == ref)) flag = true;
//		}
//		return (!flag && node->idxOfPoints.empty()); // ���node����������û�д������棬��ɾ��
//	};
//	auto isCrossSurface = [=](const size_t& nodeId, double sdf[8])->bool
//	{
//		//MXd corners(8, 3);
//		//// �õ�nodeId��Ӧ��ԭ��λ��
//		//V3d originCorner = treeOrigin + bSplineTree.getNodeCoord(nodeId, d_leafNodeWidth);
//		//corners.row(0) = originCorner;
//		//for (int i = 1; i < 8; i++)
//		//{
//		//	const int xOffset = i & 1;
//		//	const int yOffset = (i >> 1) & 1;
//		//	const int zOffset = (i >> 2) & 1;
//		//	V3d offsetCorner = originCorner + V3d(xOffset * d_leafNodeWidth.x(), yOffset * d_leafNodeWidth.y(), zOffset * d_leafNodeWidth.z());
//		//	corners.row(i) = offsetCorner;
//		//}
//		//VXd S, B;
//		//{
//		//	VXi I;
//		//	MXd C, N;
//		//	igl::signed_distance(corners, m_V, m_F, igl::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER, S, I, C, N);
//		//}
//		//sdf[0] = S(0);
//		//bool ref = (sdf[0]) > 0;
//		//bool flag = false;
//		//for (int i = 1; i < 8; i++)
//		//{
//		//	sdf[i] = S(i);
//		//	if (!flag && ((sdf[i] < 0) == ref)) flag = true;
//		//}
//
//		// �õ�nodeId��Ӧ��ԭ��λ��
//		V3d originCorner = treeOrigin + bSplineTree.getNodeCoord(nodeId, d_leafNodeWidth);
//		sdf[0] = getSignedDistance(originCorner, scene);
//		bool ref = sdf[0] > 0;
//		bool flag = false;
//		for (int i = 1; i < 8; i++)
//		{
//			const int xOffset = i & 1;
//			const int yOffset = (i >> 1) & 1;
//			const int zOffset = (i >> 2) & 1;
//			V3d offsetCorner = originCorner + V3d(xOffset * d_leafNodeWidth.x(), yOffset * d_leafNodeWidth.y(), zOffset * d_leafNodeWidth.z());
//
//			sdf[i] = getSignedDistance(offsetCorner, scene);
//			if (!flag && ((sdf[i] < 0) == ref)) flag = true;
//		}
//		return flag;
//	};
//
//	std::erase_if(d_leafNodes, isNodeCrossSurface);
//
//	std::queue<size_t> q;
//	for (auto node : d_leafNodes)
//	{
//		//if (fabs(node->boundary.first.x() + 34.6) < 1e-3 &&
//		//	fabs(node->boundary.first.y() + 44.7991) < 1e-3 &&
//		//	fabs(node->boundary.first.z() - 16.5) < 1e-3
//		//	/* &&
//		//	node->boundary.second == V3d(31.400000000, -28.299094000, 0.000000000)*/)
//		//{
//		//	cout << node->boundary.second.transpose() << endl;
//		//	cout << node->id << endl;
//		//}
//		/*if (node->id == 73)
//		{
//			cout << node->boundary.first.transpose() << endl;
//			cout << node->boundary.second.transpose() << endl;
//		}*/
//		q.push(node->id);
//	}
//
//	enum CROSS_EDGE
//	{
//		TO_RIGHT,
//		TO_LEFT,
//
//		TO_DOWN,
//		TO_UP,
//
//		TO_FRONT,
//		TO_BACK,
//	};
//
//	auto getCrossEdgeCost = [=](const int& i, const int& crossEdgeKind, const int& subDepth)->long long
//	{
//		long long val = (crossEdgeKind & 1) ? -1 : 1;
//		if (i == 0) val *= EDGE_VAL_LR[subDepth];
//		else if (i == 1) val *= EDGE_VAL_BF[subDepth];
//		else if (i == 2) val *= EDGE_VAL_BU[subDepth];
//		return val;
//	};
//
//	auto getNeightborNodeId = [&](const int& offset, const size_t& queryNodeId, size_t neighbors[6])
//	{
//		int t_offset = offset;
//
//		int fatherId = GET_PARENT_ID(queryNodeId);
//		//if (!fatherId) return; // ��֤queryNode����λ�ڵ�2�㼰����
//		int fatherOffset = GET_OFFSET(fatherId);
//		for (int i = 0; i < 3; ++i, t_offset >>= 1)
//		{
//			int queryBit = ((t_offset & 1) + 1) % 2;
//			int t_fatherId = fatherId, t_fatherOffset = fatherOffset;
//			int subDepth = 1;
//
//			// ��queryNode��ĳ�����ڵ�fatherNode���丸�ڵ��е�ƫ����fatherOffset�ĵ�iλ��queryBit(��ʾ��queryNode��Ѱ�ҵ���Χ�ڵ�ķ�����)һ��ʱ
//			// ���ǾͿ��Լ����Ѱ�ҵ�queryNode��Χ�ڵ�id��
//			while ((t_fatherOffset >> i & 1) != queryBit)
//			{
//				t_fatherId = GET_PARENT_ID(t_fatherId);
//				if (!t_fatherId) { subDepth = 0; break; }
//
//				t_fatherOffset = GET_OFFSET(t_fatherId);
//				++subDepth;
//			}
//
//			CROSS_EDGE kind = CROSS_EDGE(i * 2 + queryBit);
//			if (subDepth >= 1) neighbors[i] = queryNodeId + getCrossEdgeCost(i, kind, subDepth);
//			neighbors[i + 3] = queryNodeId + getCrossEdgeCost(i, (kind + 1) % 2 + i * 2, 0);
//		}
//	};
//
//	// ����
//	while (!q.empty())
//	{
//		auto queryNodeId = q.front();
//		q.pop();
//
//		if (visNodeId[queryNodeId] >= 2) continue;
//		visNodeId[queryNodeId]++;
//
//		// �õ����ڸ���������һ��ƫ�ƽڵ�
//		int offset = GET_OFFSET(queryNodeId);
//		// �õ���Ҫ����������nodeId(ͨ���Ҹ����Լ���ֵ)
//		size_t neighbors[6] = { 0 };
//		getNeightborNodeId(offset, queryNodeId, neighbors);
//
//		// �ж�needNode�Ƿ��������Ƿ񴩹����棬��û�����Ҵ�����������
//		for (const auto& nodeId : neighbors)
//		{
//			/*if (queryNodeId == 124)
//				cout << "neighbor: " << nodeId << endl;*/
//				// nodeId == 0 ���� nodeId ����ھӽڵ㲢������
//				// visNodeId[nodeId] >= 1�����Ѿ��ڶ����л����Ѿ���������
//			if (!nodeId || visNodeId[nodeId] >= 1) continue;
//			visNodeId[nodeId]++;
//			q.push(nodeId);
//
//			//if (nodeId == 131) 
//			//	cout << 111 << endl;
//			double sdf[8];
//			if (!isCrossSurface(nodeId, sdf)) continue;
//
//			//// create node
//			//vector<size_t> parents;
//			//int parentNodeId = GET_PARENT_ID(queryNodeId);
//			//// �����ϱ��������ҵ�����û�����ĸ��ڵ��Լ���һ�����������ĸ��ڵ�
//			//while (id2Node[parentNodeId] == nullptr && parentNodeId != 0)
//			//{
//			//	parents.emplace_back(parentNodeId);
//			//	parentNodeId = GET_PARENT_ID(parentNodeId);
//			//}
//			//parents.emplace_back(parentNodeId);
//			//// �������½����ڵ�
//			//for (auto it = parents.rbegin(); it != parents.rend(); ++it)
//			//{
//			//	const size_t parentId = *it;
//			//	OctreeNode* node = id2Node[parentId];
//			//	/*if (node == nullptr)
//			//	{
//			//		printf("parentId = %lld\n", parentId);
//			//	}*/
//			//	node->isLeaf = false;
//			//
//			//	const V3d b_beg = node->boundary.first;
//			//	const V3d b_end = node->boundary.second;
//			//	const V3d childWidth = (b_end - b_beg) * 0.5;
//			//	const size_t id = node->id;
//			//	const int depth = node->depth;
//			//
//			//	node->childs.resize(8);
//			//	for (int i = 0; i < 8; i++)
//			//	{
//			//		const int xOffset = i & 1;
//			//		const int yOffset = (i >> 1) & 1;
//			//		const int zOffset = (i >> 2) & 1;
//			//		V3d childBeg = b_beg + V3d(xOffset * childWidth.x(), yOffset * childWidth.y(), zOffset * childWidth.z());
//			//		V3d childEnd = childBeg + childWidth;
//			//		PV3d childBoundary = { childBeg, childEnd };
//			//
//			//		const size_t childId = GET_CHILD_ID(id, i);
//			//		node->childs[i] = new OctreeNode(childId, depth + 1, childWidth, childBoundary);
//			//		node->childs[i]->parent = node;
//			//		//node->childs[i]->setCorners();
//			//		id2Node[childId] = node->childs[i];
//			//
//			//		/*nLeafNodes++;
//			//		leafNodes.emplace_back(node->childs[i]);
//			//
//			//		nAllNodes++;
//			//		allNodes.emplace_back(node->childs[i]);*/
//			//	}
//			//}
//			V3d originCorner = treeOrigin + bSplineTree.getNodeCoord(nodeId, d_leafNodeWidth);
//			OctreeNode* node = new OctreeNode(nodeId, bSplineTree.maxDepth - 1, d_leafNodeWidth, { originCorner, originCorner + d_leafNodeWidth });
//			id2Node[nodeId] = node;
//			// copy sdf
//			memcpy(id2Node[nodeId]->sdf, sdf, sizeof(double) * 8);
//			id2Node[nodeId]->setEdges();
//			id2Node[nodeId]->setCorners();
//			d_leafNodes.emplace_back(id2Node[nodeId]);
//		}
//
//		//if (queryNodeId == 334)
//		//{
//		//	// create node
//		//	vector<size_t> parents;
//		//	int parentNodeId = GET_PARENT_ID(queryNodeId);
//		//	// �����ϱ��������ҵ�����û�����ĸ��ڵ��Լ���һ�����������ĸ��ڵ�
//		//	while (parentNodeId != 0)
//		//	{
//		//		parents.emplace_back(parentNodeId);
//		//		parentNodeId = GET_PARENT_ID(parentNodeId);
//		//	}
//		//	parents.emplace_back(parentNodeId);
//
//		//	for (auto parent : parents)
//		//		cout << "parent: " << parent << endl;
//		//}
//	}
//}