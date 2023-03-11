#include "MyOctree.h"
#include "utils/common.hpp"
#include <queue>
#include <Eigen\Sparse>
#include <Windows.h>
#include <numeric>
#include <chrono>

inline double BaseFunction(double x, double width, double node_x)
{
	double res = 0.0;
	if (x <= node_x - width || x >= node_x + width) return res;
	if (x <= node_x) return (1 + (x - node_x) / width);
	if (x > node_x) return (1 - (x - node_x) / width);
	return res; // in case of thoughtless
}

inline double dBaseFunction(double x, double width, double node_x)
{
	double res = 0.0;
	if (x <= node_x - width || x >= node_x + width) return res;
	if (x < node_x) return (1 / width);
	if (x > node_x) return (-1 / width);
	return res; // in case of thoughtless
}

void MyOctree::createOctree(const double& scaleSize)
{
	const size_t pointNum = modelVerts.size();
	vector<size_t> idxOfPoints(pointNum);
	std::iota(idxOfPoints.begin(), idxOfPoints.end(), 0); // 0~pointNum-1

	std::pair<V3d, V3d> boundary;
	V3d maxV = m_V.colwise().maxCoeff();
	V3d minV = m_V.colwise().minCoeff();

	// 保证外围格子是空的
	// bounding box
	V3d b_beg = minV - (maxV - minV) * scaleSize;
	V3d b_end = maxV + (maxV - minV) * scaleSize;
	V3d width = b_end - b_beg;
	boundary.first = b_beg;
	boundary.second = b_end;

	root = new OctreeNode(0, width, boundary, idxOfPoints);
	createNode(root, 0, width, boundary, idxOfPoints);

	saveNodeCorners2OBJFile("./vis/octree.obj");
}

void MyOctree::createNode(OctreeNode*& node, const int& depth, const V3d& width, const std::pair<V3d, V3d>& boundary, const vector<size_t>& idxOfPoints)
{
	vector<vector<size_t>> childPointsIdx(8);

	V3d b_beg = boundary.first;
	V3d b_end = boundary.second;
	V3d center = (b_beg + b_end) / 2.0;

	nodeXEdges[depth].emplace_back(std::make_pair(V3d(b_beg.x(), center.y(), center.z()), V3d(b_end.x(), center.y(), center.z())));
	nodeYEdges[depth].emplace_back(std::make_pair(V3d(center.x(), b_beg.y(), center.z()), V3d(center.x(), b_end.y(), center.z())));
	nodeZEdges[depth].emplace_back(std::make_pair(V3d(center.x(), center.y(), b_beg.z()), V3d(center.x(), center.y(), b_end.z())));

	nodeXEdges[depth].emplace_back(std::make_pair(V3d(b_beg.x(), center.y(), b_beg.z()), V3d(b_end.x(), center.y(), b_beg.z())));
	nodeXEdges[depth].emplace_back(std::make_pair(V3d(b_beg.x(), center.y(), b_end.z()), V3d(b_end.x(), center.y(), b_end.z())));
	nodeXEdges[depth].emplace_back(std::make_pair(V3d(b_beg.x(), b_beg.y(), center.z()), V3d(b_end.x(), b_beg.y(), center.z())));
	nodeXEdges[depth].emplace_back(std::make_pair(V3d(b_beg.x(), b_end.y(), center.z()), V3d(b_end.x(), b_end.y(), center.z())));

	nodeYEdges[depth].emplace_back(std::make_pair(V3d(center.x(), b_beg.y(), b_beg.z()), V3d(center.x(), b_end.y(), b_beg.z())));
	nodeYEdges[depth].emplace_back(std::make_pair(V3d(center.x(), b_beg.y(), b_end.z()), V3d(center.x(), b_end.y(), b_end.z())));
	nodeYEdges[depth].emplace_back(std::make_pair(V3d(b_beg.x(), b_beg.y(), center.z()), V3d(b_beg.x(), b_end.y(), center.z())));
	nodeYEdges[depth].emplace_back(std::make_pair(V3d(b_end.x(), b_beg.y(), center.z()), V3d(b_end.x(), b_end.y(), center.z())));

	nodeZEdges[depth].emplace_back(std::make_pair(V3d(center.x(), b_beg.y(), b_beg.z()), V3d(center.x(), b_beg.y(), b_end.z())));
	nodeZEdges[depth].emplace_back(std::make_pair(V3d(center.x(), b_end.y(), b_beg.z()), V3d(center.x(), b_end.y(), b_end.z())));
	nodeZEdges[depth].emplace_back(std::make_pair(V3d(b_beg.x(), center.y(), b_beg.z()), V3d(b_beg.x(), center.y(), b_end.z())));
	nodeZEdges[depth].emplace_back(std::make_pair(V3d(b_end.x(), center.y(), b_beg.z()), V3d(b_end.x(), center.y(), b_end.z())));

	if (depth + 1 >= maxDepth) return;
	if (idxOfPoints.empty()) return;

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

void MyOctree::selectLeafNode(OctreeNode* node)
{
	if (!node) return;

	std::queue<OctreeNode*> q;
	q.push(node);

	while (!q.empty())
	{
		auto froNode = q.front();
		q.pop();

		if (froNode->isLeaf)
		{
			leafNodes.emplace_back(froNode);
		}
		else
		{
			for (const auto& child : froNode->childs)
				if (child != nullptr) q.push(child);
		}
	}
}

void MyOctree::saveNodeCorners2OBJFile(const string& filename)
{
	std::ofstream out(filename);
	selectLeafNode(root);
	int count = -8;
	for (const auto& leaf : leafNodes)
	{
		vector<V3d> corners;
		corners.resize(8);
		double minX = leaf->boundary.first.x();
		double minY = leaf->boundary.first.y();
		double minZ = leaf->boundary.first.z();
		double maxX = leaf->boundary.second.x();
		double maxY = leaf->boundary.second.y();
		double maxZ = leaf->boundary.second.z();

		corners[0] = leaf->boundary.first;
		corners[1] = V3d(maxX, minY, minZ);
		corners[2] = V3d(minX, maxY, minZ);
		corners[3] = V3d(maxX, maxY, minZ);
		corners[4] = V3d(minX, minY, maxZ);
		corners[5] = V3d(maxX, minY, maxZ);
		corners[6] = V3d(minX, maxY, maxZ);
		corners[7] = leaf->boundary.second;

		for (int i = 0; i < 8; i++)
		{
			out << "v " << std::setiosflags(std::ios::fixed) << std::setprecision(9) << corners[i].x() << " " << corners[i].y() << " " << corners[i].z() << endl;
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

vector<OctreeNode*> MyOctree::getLeafNodes()
{
	if (leafNodes.size() > 2)
		return leafNodes;
	selectLeafNode(root);
	return leafNodes;
}

void MyOctree::cpIntersection()
{
	cout << "We are begining to extract edges" << endl;
	vector<V2i> modelEdges = extractEdges();

	//cout << "We are begining to select leaves" << endl;
	//selectLeafNode(root);

	// 三角形的边与node面交， 因为隐式B样条基定义在了left/bottom/back corner上， 所以与节点只需要求与这三个面的交即可

	std::cout << "开始求交\n";
	auto start = std::chrono::system_clock::now();

	cout << "边的数量：" << modelEdges.size() << endl;
	cout << "叶子节点数量：" << leafNodes.size() << endl;
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
				intersections.emplace_back(p1 + back_t * dir);
				node->isIntersectWithMesh = true;
			}
			if (isInRange(.0, 1.0, left_t) &&
				isInRange(lbbCorner.x(), lbbCorner.x() + width.x(), (p1 + left_t * dir).x()) &&
				isInRange(lbbCorner.z(), lbbCorner.z() + width.z(), (p1 + left_t * dir).z()))
			{
				intersections.emplace_back(p1 + left_t * dir);
				node->isIntersectWithMesh = true;
			}
			if (isInRange(.0, 1.0, bottom_t) &&
				isInRange(lbbCorner.x(), lbbCorner.x() + width.x(), (p1 + bottom_t * dir).x()) &&
				isInRange(lbbCorner.y(), lbbCorner.y() + width.y(), (p1 + bottom_t * dir).y()))
			{
				intersections.emplace_back(p1 + bottom_t * dir);
				node->isIntersectWithMesh = true;
			}
		}
	}
	cout << "去重前交点数量：" << intersections.size() << endl;
	//intersections.erase(std::unique(intersections.begin(), intersections.end()), intersections.end());
	//cout << "去重后交点数量：" << intersections.size() << endl;
	auto end = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	cout << "花费了"
		<< double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den
		<< "秒" << endl;
	std::cout << "#####################################################################" << endl;

	saveIntersections("./vis/edgeInter.xyz", intersections);

	// 三角形面与node边线交
	std::cout << "Continue to compute the intersections for triangle faces..." << endl;
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

		for (int i = 0; i < maxDepth; ++i)
		{
			for (const auto& edge : nodeXEdges[i])
			{
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
						intersections.emplace_back(V3d(interX, y, z));
					}
				}
			}

			for (const auto& edge : nodeYEdges[i])
			{
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
						intersections.emplace_back(V3d(x, interY, z));
					}
				}
			}

			for (const auto& edge : nodeZEdges[i])
			{
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
						intersections.emplace_back(V3d(x, y, interZ));
					}
				}
			}
		}
	}
}

void MyOctree::saveIntersections(const string& filename, const vector<V3d>& intersections) const
{
	std::ofstream out(filename);
	for (V3d p : intersections)
	{
		//out << "v " << p.x() << " " << p.y() << " " << p.z() << endl;
		out << p.x() << " " << p.y() << " " << p.z() << endl;
	}
	/*for (V3d p : UnitNormals)
	{
		out << "vn " << p.x() << " " << p.y() << " " << p.z() << endl;
	}*/
	out.close();
}

SpMat MyOctree::coEfficientOfPoints(const vector<V3d>& edgeIntersections, const vector<V3d>& faceIntersections, SpMat& Bx, SpMat& By, SpMat& Bz)
{
	vector<V3d> allPoints;
	for (V3d p : edgeIntersections) allPoints.emplace_back(p);
	for (V3d p : faceIntersections) allPoints.emplace_back(p);
	int N_points = allPoints.size();
	int N_leafNodes = leafNodes.size();

	SpMat B(N_points, N_leafNodes);        // B = (B(x)B(y)B(z))_allPoints x leafNodes
	Bx.resize(N_points, N_leafNodes);  // Bx = (B'(x)B(y)B(z))_allPoints x leafNodes
	By.resize(N_points, N_leafNodes);  // By = (B(x)B'(y)B(z))_allPoints x leafNodes
	Bz.resize(N_points, N_leafNodes);  // Bz = (B(x)B(y)B'(z))_allPoints x leafNodes
	vector<Eigen::Triplet<double>> triplets;
	vector<Eigen::Triplet<double>> tripletsBx;
	vector<Eigen::Triplet<double>> tripletsBy;
	vector<Eigen::Triplet<double>> tripletsBz;
	for (int i = 0; i < N_points; i++)
	{
		for (int j = 0; j < N_leafNodes; j++)
		{
			V3d nodeCorner = leafNodes[j]->boundary.first;
			V3d width = leafNodes[j]->width;
			double x = BaseFunction(allPoints[i].x(), width.x(), nodeCorner.x());
			double y = BaseFunction(allPoints[i].y(), width.y(), nodeCorner.y());
			double z = BaseFunction(allPoints[i].z(), width.z(), nodeCorner.z());


			double dx = dBaseFunction(allPoints[i].x(), width.x(), nodeCorner.x());
			double dy = dBaseFunction(allPoints[i].y(), width.y(), nodeCorner.y());
			double dz = dBaseFunction(allPoints[i].z(), width.z(), nodeCorner.z());
			double w3 = width.x() * width.y() * width.z();
			dx = dx * y * z;
			if (dx != 0.0) tripletsBx.emplace_back(i, j, dx);
			dy = x * dy * z;
			if (dy != 0.0) tripletsBy.emplace_back(i, j, dy);
			dz = x * y * dz;
			if (dz != 0.0) tripletsBz.emplace_back(i, j, dz);
			double baseValue = x * y * z;
			if (baseValue != 0.0) triplets.emplace_back(i, j, baseValue);
		}
	}
	Bx.setFromTriplets(tripletsBx.begin(), tripletsBx.end());
	By.setFromTriplets(tripletsBy.begin(), tripletsBy.end());
	Bz.setFromTriplets(tripletsBz.begin(), tripletsBz.end());
	B.setFromTriplets(triplets.begin(), triplets.end());
	return B;
}

void MyOctree::saveBValue2TXT(const string& filename, const Eigen::VectorXd& X) const
{
	std::ofstream out(filename);
	out << std::setiosflags(std::ios::fixed) << std::setprecision(9) << X << endl;
	out.close();
}
