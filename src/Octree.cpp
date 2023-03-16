#include "Octree.h"
#include "SDFHelper.h"
#include "utils\common.hpp"
#include "MarchingCubes.hpp"
#include <queue>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <Windows.h>
#include <Eigen\Sparse>
#include <igl\writeOBJ.h>

void OctreeNode::setCorner()
{
	double minX = boundary.first.x();
	double minY = boundary.first.y();
	double minZ = boundary.first.z();
	double maxX = boundary.second.x();
	double maxY = boundary.second.y();
	double maxZ = boundary.second.z();

	corners[0] = boundary.first;
	corners[1] = V3d(maxX, minY, minZ);
	corners[2] = V3d(minX, maxY, minZ);
	corners[3] = V3d(maxX, maxY, minZ);
	corners[4] = V3d(minX, minY, maxZ);
	corners[5] = V3d(maxX, minY, maxZ);
	corners[6] = V3d(minX, maxY, maxZ);
	corners[7] = boundary.second;
}

void OctreeNode::setEdges()
{
	// X
	edges.emplace_back(std::make_pair(corners[0], corners[1]));
	edges.emplace_back(std::make_pair(corners[2], corners[3]));
	edges.emplace_back(std::make_pair(corners[4], corners[5]));
	edges.emplace_back(std::make_pair(corners[6], corners[7]));

	// Y
	edges.emplace_back(std::make_pair(corners[0], corners[2]));
	edges.emplace_back(std::make_pair(corners[1], corners[3]));
	edges.emplace_back(std::make_pair(corners[4], corners[6]));
	edges.emplace_back(std::make_pair(corners[5], corners[7]));

	// Z
	edges.emplace_back(std::make_pair(corners[0], corners[4]));
	edges.emplace_back(std::make_pair(corners[1], corners[5]));
	edges.emplace_back(std::make_pair(corners[2], corners[6]));
	edges.emplace_back(std::make_pair(corners[3], corners[7]));
}

void OctreeNode::setDomain()
{
	domain[0] = boundary.first - width;
	domain[1] = boundary.second;
}

bool OctreeNode::isInDomain(const OctreeNode* otherNode)
{
	V3d nodeOrigin = boundary.first;
	V3d st_domain = otherNode->domain[0];
	V3d ed_domain = otherNode->domain[1];

	if (st_domain.x() <= nodeOrigin.x() && nodeOrigin.x() < ed_domain.x() &&
		st_domain.y() <= nodeOrigin.y() && nodeOrigin.y() < ed_domain.y() &&
		st_domain.z() <= nodeOrigin.z() && nodeOrigin.z() < ed_domain.z())
		return true;
	return false;
}

inline double BaseFunction(const double& x, const double& node_x, const double& w)
{
	if (x <= node_x - w || x >= node_x + w) return 0.0;
	if (x <= node_x) return 1 + (x - node_x) / w;
	if (x > node_x) return 1 - (x - node_x) / w;
}

inline double OctreeNode::BaseFunction4Point(const V3d& p)
{
	const V3d nodePosition = boundary.first;
	double x = BaseFunction(p.x(), nodePosition.x(), width.x());
	if (x <= 0.0) return 0.0;
	double y = BaseFunction(p.y(), nodePosition.y(), width.y());
	if (y <= 0.0) return 0.0;
	double z = BaseFunction(p.z(), nodePosition.z(), width.z());
	if (z <= 0) return 0.0;
	return x * y * z;
}

void Octree::createOctree(const double& scaleSize)
{
	const size_t pointNum = modelVerts.size();
	vector<size_t> idxOfPoints(pointNum);
	std::iota(idxOfPoints.begin(), idxOfPoints.end(), 0); // 0~pointNum-1

	V3d maxV = m_V.colwise().maxCoeff();
	V3d minV = m_V.colwise().minCoeff();

	// 保证外围格子是空的
	// bounding box
	V3d b_beg = minV - (maxV - minV) * scaleSize;
	V3d b_end = maxV + (maxV - minV) * scaleSize;
	PV3d boundary = std::make_pair(b_beg, b_end);
	V3d width = b_end - b_beg;
	bb = BoundingBox(b_beg, b_end);

	root = new OctreeNode(0, width, boundary, idxOfPoints);
	createNode(root, 0, width, boundary, idxOfPoints);

	saveNodeCorners2OBJFile(concatString(DELIMITER, VIS_DIR, std::to_string(maxDepth), "octree.obj"));
}

void Octree::createNode(OctreeNode*& node, const int& depth, const V3d& width, const std::pair<V3d, V3d>& boundary, const vector<size_t>& idxOfPoints)
{
	numNodes++;
	vector<vector<size_t>> childPointsIdx(8);

	V3d b_beg = boundary.first;
	V3d b_end = boundary.second;
	V3d center = (b_beg + b_end) / 2.0;

	if (depth + 1 >= maxDepth || idxOfPoints.empty())
	{
		node->setCorner();
		node->setEdges();
		node->setDomain();
		nLeafNodes++;
		leafNodes.emplace_back(node);
		return;
	}

	node->isLeaf = false;
	for (size_t idx : idxOfPoints)
	{
		V3d p = modelVerts[idx];

		int whichChild = 0;
		if (p.x() > center.x()) whichChild |= 1;
		if (p.y() > center.y()) whichChild |= 2;
		if (p.z() > center.z()) whichChild |= 4;

		childPointsIdx[whichChild].emplace_back(idx);
	}

	node->childs.resize(8, nullptr);
	for (int i = 0; i < 8; i++)
	{
		const int xOffset = i & 1;
		const int yOffset = (i >> 1) & 1;
		const int zOffset = (i >> 2) & 1;
		V3d childWidth = width / 2.0;
		V3d childBeg = b_beg + V3d(xOffset * childWidth.x(), yOffset * childWidth.y(), zOffset * childWidth.z());
		V3d childEnd = childBeg + childWidth;
		PV3d childBoundary = { childBeg, childEnd };

		node->childs[i] = new OctreeNode(depth + 1, childWidth, childBoundary, childPointsIdx[i]);
		node->childs[i]->parent = node;

		createNode(node->childs[i], depth + 1, childWidth, childBoundary, childPointsIdx[i]);
	}
}

void Octree::cpIntersection()
{
	cout << "Extracting edges from " << std::quoted(modelName) << "..." << endl;
	vector<V2i> modelEdges = extractEdges();
	cout << "--Number of edges = " << modelEdges.size() << endl;
	cout << "--Number of leaf nodes = " << nLeafNodes << endl;

	// 三角形的边与node面交， 因为隐式B样条基定义在了left/bottom/back corner上， 所以与节点只需要求与这三个面的交即可
	std::cout << "Continue to compute the intersections for triangle EDGES...\n";
	auto start = std::chrono::system_clock::now();

	for (int i = 0; i < modelEdges.size(); i++)
	{
		Eigen::Vector2i e = modelEdges[i];
		V3d p1 = m_V.row(e.x());
		V3d p2 = m_V.row(e.y());
		V3d dir = p2 - p1;

		for (auto node : leafNodes)
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
				interPoints.emplace_back(p1 + back_t * dir);
				node->isInterMesh = true;
			}
			if (isInRange(.0, 1.0, left_t) &&
				isInRange(lbbCorner.x(), lbbCorner.x() + width.x(), (p1 + left_t * dir).x()) &&
				isInRange(lbbCorner.z(), lbbCorner.z() + width.z(), (p1 + left_t * dir).z()))
			{
				interPoints.emplace_back(p1 + left_t * dir);
				node->isInterMesh = true;
			}
			if (isInRange(.0, 1.0, bottom_t) &&
				isInRange(lbbCorner.x(), lbbCorner.x() + width.x(), (p1 + bottom_t * dir).x()) &&
				isInRange(lbbCorner.y(), lbbCorner.y() + width.y(), (p1 + bottom_t * dir).y()))
			{
				interPoints.emplace_back(p1 + bottom_t * dir);
				node->isInterMesh = true;
			}
		}
	}

	auto end = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	cout << "三角形边与node的交点数量：" << interPoints.size() <<
		", spent " << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << "s." << endl;
	//intersections.erase(std::unique(intersections.begin(), intersections.end()), intersections.end());
	std::cout << "#####################################################################" << endl;
	saveIntersections("./vis/4/edgeInter.xyz", interPoints);

	// 三角形面与node边线交（有重合点）
	start = std::chrono::system_clock::now();
	vector<V3d> faceIntersections;
	std::cout << "Continue to compute the intersections for triangle FACES..." << endl;

	for (const auto& leafNode : leafNodes)
	{
		auto edges = leafNode->edges;

		for (int i = 0; i < modelFaces.size(); i++)
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
						interPoints.emplace_back(V3d(interX, y, z));
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
						interPoints.emplace_back(V3d(x, interY, z));
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
						interPoints.emplace_back(V3d(x, y, interZ));
						faceIntersections.emplace_back(V3d(x, y, interZ));
						leafNode->isInterMesh = true;
					}
				}
			}
		}

		if (leafNode->isInterMesh) interLeafNodes.emplace_back(leafNode); // 筛选有交点的叶子节点
	}
	end = std::chrono::system_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	faceIntersections.erase(std::unique(faceIntersections.begin(), faceIntersections.end()), faceIntersections.end());
	cout << "三角形面与node边的交点数量：" << faceIntersections.size() <<
		", spent " << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << "s." << endl;

	saveIntersections("./vis/4/faceInter.xyz", faceIntersections);

	interPoints.erase(std::unique(interPoints.begin(), interPoints.end()), interPoints.end());
	cout << "总交点数量：" << interPoints.size() << endl;
}

void Octree::setInDomainLeafNode()
{
	if (interLeafNodes.empty()) return;
	inDomainLeafNodes.resize(interLeafNodes.size());
	for (int i = 0; i < interLeafNodes.size(); ++i)
	{
		for (const auto& leafNode : leafNodes)
		{
			auto iNode = interLeafNodes[i];
			if (iNode->isInDomain(leafNode))
				inDomainLeafNodes[i].emplace_back(leafNode);
		}
	}
}

//TODO: 待并行
void Octree::setSDF(std::string& filename)
{
	// initialize a 3d scene
	fcpw::Scene<3> scene;
	initSDF(scene, modelVerts, modelFaces);

	//std::ofstream out("./output/4/SDFValue.txt");
	std::ofstream out(filename);

	int cnt = 1;
	for (auto& leaf : leafNodes)
	{
		double minX = leaf->boundary.first.x();
		double minY = leaf->boundary.first.y();
		double minZ = leaf->boundary.first.z();
		double maxX = leaf->boundary.second.x();
		double maxY = leaf->boundary.second.y();
		double maxZ = leaf->boundary.second.z();

		leaf->corners[0] = leaf->boundary.first;
		leaf->corners[1] = V3d(maxX, minY, minZ);
		leaf->corners[2] = V3d(minX, maxY, minZ);
		leaf->corners[3] = V3d(maxX, maxY, minZ);
		leaf->corners[4] = V3d(minX, minY, maxZ);
		leaf->corners[5] = V3d(maxX, minY, maxZ);
		leaf->corners[6] = V3d(minX, maxY, maxZ);
		leaf->corners[7] = leaf->boundary.second;

		out << cnt++ << endl;
		for (int i = 0; i < 8; ++i)
		{
			leaf->sdfVal[i] = getSignedDistance(leaf->corners[i], scene);
			out << leaf->sdfVal[i] << endl;
		}
	}

	out.close();
}

void Octree::setBSplineValue(std::string& filename)
{
	const size_t nVerts = modelVerts.size();
	const size_t nInterPoints = interPoints.size(); // 交点数量
	BSplineValue.resize(nVerts + nInterPoints);
	BSplineValue.setZero();

	const size_t nLeafNodes = leafNodes.size();
	const size_t nInterLeafNodes = interLeafNodes.size();

	double output = 100.0 / (nVerts + nInterPoints);

	std::ofstream out(filename);

	for (int i = 0; i < nVerts; ++i)
	{
		std::cout << "\r Computing[" << std::fixed << std::setprecision(2) << i * output << "%]";
		out << i + 1 << endl;
		V3d modelPoint = modelVerts[i];

		for (int j = 0; j < nInterLeafNodes; ++j)
		{
			for (int k = 0; k < inDomainLeafNodes[j].size(); ++k)
			{
				auto node = inDomainLeafNodes[j][k];

				out << node->BaseFunction4Point(modelPoint) << endl;
				BSplineValue[i] += node->sdfVal[0] * (node->BaseFunction4Point(modelPoint));
			}
		}
	}

	int cnt = 0;
	for (int i = 0; i < nInterPoints; ++i)
	{
		cnt = i + nVerts;
		out << cnt + 1 << endl;
		std::cout << "\r Computing[" << std::fixed << std::setprecision(2) << cnt * output << "%]";
		V3d interPoint = interPoints[i];

		for (int j = 0; j < nInterLeafNodes; ++j)
		{
			for (int k = 0; k < inDomainLeafNodes[j].size(); ++k)
			{
				auto node = inDomainLeafNodes[j][k];

				out << node->BaseFunction4Point(interPoint) << endl;
				BSplineValue[cnt] += node->sdfVal[0] * (node->BaseFunction4Point(interPoint));
			}
		}
	}

	out.close();
}

inline void Octree::saveNodeCorners2OBJFile(const string& filename) const
{
	std::ofstream out(filename);
	int count = -8;
	for (const auto& leaf : leafNodes)
	{
		for (int i = 0; i < 8; i++)
		{
			out << "v " << std::setiosflags(std::ios::fixed) << std::setprecision(9) << leaf->corners[i].x() << " " << leaf->corners[i].y() << " " << leaf->corners[i].z() << endl;
			count++;
		}
		out << "l " << 1 + count << " " << 2 + count << endl;
		out << "l " << 1 + count << " " << 3 + count << endl;
		out << "l " << 1 + count << " " << 5 + count << endl;
		out << "l " << 2 + count << " " << 4 + count << endl;
		out << "l " << 2 + count << " " << 6 + count << endl;
		out << "l " << 3 + count << " " << 4 + count << endl;
		out << "l " << 3 + count << " " << 7 + count << endl;
		out << "l " << 4 + count << " " << 8 + count << endl;
		out << "l " << 5 + count << " " << 6 + count << endl;
		out << "l " << 5 + count << " " << 7 + count << endl;
		out << "l " << 6 + count << " " << 8 + count << endl;
		out << "l " << 7 + count << " " << 8 + count << endl;
	}
	out.close();
}

inline void Octree::saveIntersections(const string& filename, const vector<V3d>& intersections) const
{
	std::ofstream out(filename);
	for (V3d p : intersections)
		out << p.x() << " " << p.y() << " " << p.z() << endl;
	out.close();
}

inline void Octree::saveBSplineValue(const string& filename) const
{
	std::ofstream out(filename);
	out << std::setiosflags(std::ios::fixed) << std::setprecision(9) << BSplineValue << endl;
	out.close();
}

inline void Octree::mcVisualization(const string& filename, const V3i& resolution) const
{
	auto mesh = MC::extractIsoSurface(bb.boxOrigin, bb.boxWidth, resolution, [&](const V3d& voxelVert)->double {
		double sum = 0.0;
		for (const auto& node : leafNodes)
			sum += node->sdfVal[0] * (node->BaseFunction4Point(voxelVert));
		return sum;
		});

	MXd verts;
	MXi faces;
	list2Matrix<MXd, double, 3>(mesh.meshVerts, verts);
	list2Matrix<MXi, int, 3>(mesh.meshFaces, faces);
	igl::writeOBJ(filename, verts, faces);
}

inline void Octree::txtVisualization(const string& filename) const
{
	writeTexturedObjFile(filename, BSplineValue);
}