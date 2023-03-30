#include "ThinShells.h"
#include "SDFHelper.h"
#include "BSpline.hpp"
#include "utils\common.hpp"
#include "utils\cuda\CUDAMath.hpp"
#include "cuAcc\MarchingCubes\MarchingCubes.h"

//////////////////////
//  Create  Shells  //
//////////////////////
inline void ThinShells::cpIntersectionPoints()
{
	vector<V2i> modelEdges = extractEdges();
	uint nModelEdges = modelEdges.size();
	cout << "--Number of leaf nodes = " << bSplineTree.nLeafNodes << endl;;

	// 三角形的边与node面交， 因为隐式B样条基定义在了left/bottom/back corner上， 所以与节点只需要求与这三个面的交即可
	std::cout << "Compute the intersections between triangle EDGES and nodes...\n";
	for (int i = 0; i < nModelEdges; i++)
	{
		Eigen::Vector2i e = modelEdges[i];
		V3d p1 = m_V.row(e.x());
		V3d p2 = m_V.row(e.y());
		V3d dir = p2 - p1;

		for (auto node : bSplineTree.leafNodes)
		{
			V3d lbbCorner = node->boundary.first;
			V3d width = node->width;

			// back plane
			double back_t = DINF;
			if (dir.x() != 0)
				back_t = (lbbCorner.x() - p1.x()) / dir.x();
			// left plane
			double left_t = DINF;
			if (dir.y() != 0)
				left_t = (lbbCorner.y() - p1.y()) / dir.y();
			// bottom plane
			double bottom_t = DINF;
			if (dir.z() != 0)
				bottom_t = (lbbCorner.z() - p1.z()) / dir.z();

			if (isInRange(.0, 1.0, back_t) &&
				isInRange(lbbCorner.y(), lbbCorner.y() + width.y(), (p1 + back_t * dir).y()) &&
				isInRange(lbbCorner.z(), lbbCorner.z() + width.z(), (p1 + back_t * dir).z()))
			{
				bSplineTree.edgeInterPoints.emplace_back(p1 + back_t * dir);
				node->isInterMesh = true;
			}
			if (isInRange(.0, 1.0, left_t) &&
				isInRange(lbbCorner.x(), lbbCorner.x() + width.x(), (p1 + left_t * dir).x()) &&
				isInRange(lbbCorner.z(), lbbCorner.z() + width.z(), (p1 + left_t * dir).z()))
			{
				bSplineTree.edgeInterPoints.emplace_back(p1 + left_t * dir);
				node->isInterMesh = true;
			}
			if (isInRange(.0, 1.0, bottom_t) &&
				isInRange(lbbCorner.x(), lbbCorner.x() + width.x(), (p1 + bottom_t * dir).x()) &&
				isInRange(lbbCorner.y(), lbbCorner.y() + width.y(), (p1 + bottom_t * dir).y()))
			{
				bSplineTree.edgeInterPoints.emplace_back(p1 + bottom_t * dir);
				node->isInterMesh = true;
			}
		}
	}
	cout << "--三角形边与node的交点数量：" << bSplineTree.edgeInterPoints.size() << endl;
	bSplineTree.allInterPoints.insert(bSplineTree.allInterPoints.end(), bSplineTree.edgeInterPoints.begin(), bSplineTree.edgeInterPoints.end());

	// 三角形面与node边线交（有重合点）
	vector<V3d> faceIntersections;
	std::cout << "Compute the intersections between triangle FACES and node EDGES..." << endl;
	for (const auto& leafNode : bSplineTree.leafNodes)
	{
		auto edges = leafNode->edges;

		for (int i = 0; i < nModelEdges; i++)
		{
			V3i f = modelFaces[i];
			V3d p1 = modelVerts[f.x()];
			V3d p2 = modelVerts[f.y()];
			V3d p3 = modelVerts[f.z()];
			Eigen::Matrix3d faceMatrix;
			faceMatrix << p1.x(), p2.x(), p3.x(),
				p1.y(), p2.y(), p3.y(),
				p1.z(), p2.z(), p3.z();
			V3d maxElement = faceMatrix.rowwise().maxCoeff();
			V3d minElement = faceMatrix.rowwise().minCoeff();

			Triangle t(p1, p2, p3);

			for (int j = 0; j < 4; ++j)
			{
				auto edge = edges[j];
				double y = edge.first.y();
				double z = edge.first.z();
				if (maxElement.x() <= edge.first.x() || minElement.x() >= edge.second.x()) continue;
				if (minElement.y() >= y || maxElement.y() <= y || minElement.z() >= z || maxElement.z() <= z) continue;
				V3d coefficient = t.cpCoefficientOfTriangle(y, z, 1);
				if (coefficient.x() > 0)
				{
					double interX = coefficient.x() * p1.x() + coefficient.y() * p2.x() + coefficient.z() * p3.x();
					if (interX >= edge.first.x() && interX <= edge.second.x())
					{
						faceIntersections.emplace_back(V3d(interX, y, z));
						leafNode->isInterMesh = true;
					}
				}
			}

			for (int j = 4; j < 8; ++j)
			{
				auto edge = edges[j];
				double x = edge.first.x();
				double z = edge.first.z();
				if (maxElement.y() <= edge.first.y() || minElement.y() >= edge.second.y()) continue;
				if (minElement.x() >= x || maxElement.x() <= x || minElement.z() >= z || maxElement.z() <= z) continue;
				V3d coefficient = t.cpCoefficientOfTriangle(z, x, 2);
				if (coefficient.x() > 0)
				{
					double interY = coefficient.x() * p1.y() + coefficient.y() * p2.y() + coefficient.z() * p3.y();
					if (interY >= edge.first.y() && interY <= edge.second.y())
					{
						faceIntersections.emplace_back(V3d(x, interY, z));
						leafNode->isInterMesh = true;
					}
				}
			}

			for (int j = 8; j < 11; ++j)
			{
				auto edge = edges[j];
				double x = edge.first.x();
				double y = edge.first.y();
				if (maxElement.z() <= edge.first.z() || minElement.z() >= edge.second.z()) continue;
				if (minElement.x() >= x || maxElement.x() <= x || minElement.y() >= y || maxElement.y() <= y) continue;
				V3d coefficient = t.cpCoefficientOfTriangle(x, y, 0);
				if (coefficient.x() > 0)
				{
					double interZ = coefficient.x() * p1.z() + coefficient.y() * p2.z() + coefficient.z() * p3.z();
					if (interZ >= edge.first.z() && interZ <= edge.second.z())
					{
						faceIntersections.emplace_back(V3d(x, y, interZ));
						leafNode->isInterMesh = true;
					}
				}
			}
		}

		if (leafNode->isInterMesh) bSplineTree.interLeafNodes.emplace_back(leafNode); // 筛选有交点的叶子节点
	}
	faceIntersections.erase(std::unique(faceIntersections.begin(), faceIntersections.end()), faceIntersections.end());
	cout << "--三角形面与node边的交点数量：" << faceIntersections.size() << endl;
	bSplineTree.allInterPoints.insert(bSplineTree.allInterPoints.end(), bSplineTree.faceInterPoints.begin(), bSplineTree.faceInterPoints.end());

	bSplineTree.allInterPoints.erase(std::unique(bSplineTree.allInterPoints.begin(), bSplineTree.allInterPoints.end()), bSplineTree.allInterPoints.end());
	cout << "--总交点数量：" << bSplineTree.allInterPoints.size() << endl;
}

inline void ThinShells::cpSDFOfTreeNodes()
{
	const uint nAllNodes = bSplineTree.numNodes;

	// initialize a 3d scene
	fcpw::Scene<3> scene;
	initSDF(scene, modelVerts, modelFaces);

	bSplineTree.sdfVal.resize(nAllNodes * 8);
	//sdfVal.resize(nLeafNodes * 8);

	for (int i = 0; i < nAllNodes; ++i)
		//for (int i = 0; i < nLeafNodes; ++i)
		for (int k = 0; k < 8; ++k)
			bSplineTree.sdfVal(i * 8 + k) = getSignedDistance(bSplineTree.allNodes[i]->corners[k], scene);
}

inline void ThinShells::cpCoefficients()
{
	using SpMat = Eigen::SparseMatrix<double>;
	using Trip = Eigen::Triplet<double>;

	const uint numNodes = bSplineTree.numNodes;

	// initial matrix
	SpMat sm(numNodes * 8, numNodes * 8); // A
	vector<Trip> matVal;

	for (int i = 0; i < numNodes; ++i)
	{
		auto node_i = bSplineTree.allNodes[i];
		//auto node_i = leafNodes[i];
		map<size_t, bool> visID;
		auto [inDmPoints, inDmPointsID] = bSplineTree.setInDomainPoints(node_i, visID);
		const int nInDmPoints = inDmPoints.size();

		for (int k = 0; k < 8; ++k)
		{
			V3d i_corner = node_i->corners[k];
			const uint ic_row = i * 8 + k;

			for (const auto& id_ck : bSplineTree.corner2IDs[i_corner])
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
				assert(inDmPointsID[j] < numNodes * 8, "index of col > numNodes * 8!");
				if (val != 0) matVal.emplace_back(Trip(ic_row, inDmPointsID[j], val));
			}
		}
	}
	sm.setFromTriplets(matVal.begin(), matVal.end());
	//sm.makeCompressed();

	auto A = sm;
	auto b = bSplineTree.sdfVal;
	/*auto A = sm.transpose() * sm;
	auto b = sm.transpose() * sdfVal;*/

	Eigen::LeastSquaresConjugateGradient<SpMat> lscg;
	lscg.compute(A);
	bSplineTree.lambda = lscg.solve(b);
	cout << "Residual Error: " << (A * (bSplineTree.lambda) - b).norm() << endl;

	//saveCoefficients(concatFilePath((string)OUT_DIR, modelName, std::to_string(maxDepth), (string)"Coefficients.txt"));
}

inline void ThinShells::cpBSplineValue()
{
	const uint numNodes = bSplineTree.numNodes;

	const uint nInterPoints = bSplineTree.allInterPoints.size();
	const uint nInterLeafNodes = bSplineTree.interLeafNodes.size();

	bSplineTree.BSplineValue.resize(nModelVerts + nInterPoints);
	bSplineTree.BSplineValue.setZero();

	for (int i = 0; i < nModelVerts; ++i)
	{
		V3d modelPoint = modelVerts[i];

		for (int j = 0; j < numNodes; ++j)
		{
			auto node = bSplineTree.allNodes[j];
			for (int k = 0; k < 8; k++)
				bSplineTree.BSplineValue[i] += bSplineTree.lambda[j * 8 + k] * (BaseFunction4Point(node->corners[k], node->width, modelPoint));
		}
	}

	int cnt = 0;
	for (int i = 0; i < nInterPoints; ++i)
	{
		cnt = i + nModelVerts;
		V3d interPoint = bSplineTree.allInterPoints[i];

		for (int j = 0; j < numNodes; ++j)
		{
			auto node = bSplineTree.allNodes[j];
			for (int k = 0; k < 8; k++)
				bSplineTree.BSplineValue[cnt] += bSplineTree.lambda[j * 8 + k] * (BaseFunction4Point(node->corners[k], node->width, interPoint));
		}
	}

	innerShellIsoVal = *(std::min_element(bSplineTree.BSplineValue.begin(), bSplineTree.BSplineValue.end()));
	outerShellIsoVal = *(std::max_element(bSplineTree.BSplineValue.begin(), bSplineTree.BSplineValue.end()));

	//bSplineTree.saveBSplineValue(concatFilePath((string)OUT_DIR, modelName, std::to_string(treeDepth), (string)"BSplineValue.txt"));
}

inline void ThinShells::initBSplineTree()
{
	cpIntersectionPoints();

	cpSDFOfTreeNodes();

	cpCoefficients();

	cpBSplineValue();
}

void ThinShells::creatShell()
{
	initBSplineTree();
}

//////////////////////
//   Visualiztion   //
//////////////////////
void ThinShells::mcVisualization(const string& innerFilename, const V3i& innerResolution,
	const string& outerFilename, const V3i& outerResolution) const
{
	V3d gridOrigin = modelBoundingBox.boxOrigin;
	V3d gridWidth = modelBoundingBox.boxWidth;

	if (!innerFilename.empty())
		MC::marching_cubes(bSplineTree.allNodes, bSplineTree.lambda, make_double3(gridOrigin), make_double3(gridWidth),
			make_uint3(innerResolution), innerShellIsoVal, innerFilename);

	if (!outerFilename.empty())
		MC::marching_cubes(bSplineTree.allNodes, bSplineTree.lambda, make_double3(gridOrigin), make_double3(gridWidth),
			make_uint3(outerResolution), outerShellIsoVal, outerFilename);
}

void ThinShells::textureVisualization(const string& filename) const
{
	bSplineTree.textureVisualization(filename);
}

void ThinShells::saveInterPoints(string& filename_1, string& filename_2)
{
	if (filename_1.empty()) filename_1 = concatFilePath((string)VIS_DIR, modelName, std::to_string(treeDepth), (string)"edgeInter.xyz");
	bSplineTree.saveIntersections(filename_1, bSplineTree.edgeInterPoints);

	if (filename_2.empty()) filename_2 = concatFilePath((string)VIS_DIR, modelName, std::to_string(treeDepth), (string)"faceInter.xyz");
	bSplineTree.saveIntersections(filename_2, bSplineTree.faceInterPoints);
}

void ThinShells::SaveBValue2TXT(string& filename, VXd X)
{
	std::ofstream out(filename);
	out << std::setiosflags(std::ios::fixed) << std::setprecision(9) << X << std::endl;
	out.close();
}
