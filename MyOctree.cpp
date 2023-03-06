#include "MyOctree.h"
#include <queue>
#include <Eigen\Sparse>
#include <Windows.h>
#include <numeric>

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
	boundary.first = minV;
	boundary.second = maxV;

	// 保证外围格子是空的
	// bounding box
	V3d b_beg = minV - (maxV - minV) * scaleSize;
	V3d b_end = maxV + (maxV - minV) * scaleSize;
	V3d width = V3d((b_end - b_beg)(0), (b_end - b_beg)(1), (b_end - b_beg)(2));

	createNode(root, nullptr, 0, width, idxOfPoints, boundary);
}

void MyOctree::createNode(OctreeNode*& node, OctreeNode* parent, const int& depth, const V3d& width, const vector<size_t>& idxOfPoints, const std::pair<V3d, V3d>& boundary)
{
	if (depth >= maxDepth || idxOfPoints.empty()) return;

	node = new OctreeNode(depth, width, boundary, idxOfPoints);
	node->parent = parent;

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

	if (depth + 1 == maxDepth) return;

	bool hasChild = false;
	for (size_t idx : idxOfPoints)
	{
		V3d p = modelVerts[idx];

		int whichChild = 0;
		if (p.x() > center.x()) whichChild |= 1;
		if (p.y() > center.y()) whichChild |= 2;
		if (p.z() > center.z()) whichChild |= 4;

		childPointsIdx[whichChild].emplace_back(idx);
		if (!childPointsIdx[whichChild].empty() && !hasChild) hasChild = true;
	}
	if (hasChild) node->isLeaf = false;
	else return;

	for (int i = 0; i < 8; i++)
	{
		const int xOffset = i & 1;
		const int yOffset = (i >> 1) & 1;
		const int zOffset = (i >> 2) & 1;
		V3d childBeg = b_beg + V3d(xOffset * width.x(), yOffset * width.y(), zOffset * width.z());
		V3d childEnd = childBeg + V3d(width.x(), width.y(), width.z());
		PV3d childBoundary = { childBeg, childEnd };

		createNode(node->child[i], node, depth + 1, width / 2.0, childPointsIdx[i], childBoundary);
	}
}

void MyOctree::selectLeafNode(OctreeNode* node)
{
	std::queue<OctreeNode*> q;
	q.push(node);

	while (!q.empty())
	{
		auto froNode = q.front();
		q.pop();

		if (froNode->isLeaf)
			leafNodes.emplace_back(froNode);
		else
		{
			for (int i = 0; i < 8; ++i)
				q.push(froNode->child[i]);
		}
	}
}

void MyOctree::saveNodeCorners2OBJFile(string filename)
{
	ofstream out(filename);
	selectLeafNode(root);
	int count = -8;
	for (int i = 0; i < m_leafNodes.size(); i++)
	{
		auto leaf = m_leafNodes[i];
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
			out << "v " << setiosflags(ios::fixed) << setprecision(9) << corners[i].x() << " " << corners[i].y() << " " << corners[i].z() << endl;
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

		/*out << "f " << 1 + count << " " << 2 + count << " " << 4 + count << endl;
		out << "f " << 3 + count << " " << 1 + count << " " << 4 + count << endl;
		out << "f " << 1 + count << " " << 3 + count << " " << 7 + count << endl;
		out << "f " << 1 + count << " " << 7 + count << " " << 5 + count << endl;
		out << "f " << 1 + count << " " << 5 + count << " " << 6 + count << endl;
		out << "f " << 1 + count << " " << 6 + count << " " << 2 + count << endl;
		out << "f " << 7 + count << " " << 5 + count << " " << 6 + count << endl;
		out << "f " << 7 + count << " " << 6 + count << " " << 8 + count << endl;
		out << "f " << 4 + count << " " << 3 + count << " " << 7 + count << endl;
		out << "f " << 4 + count << " " << 7 + count << " " << 8 + count << endl;
		out << "f " << 6 + count << " " << 2 + count << " " << 4 + count << endl;
		out << "f " << 6 + count << " " << 4 + count << " " << 8 + count << endl;*/
	}
	out.close();
}

vector<OctreeNode*> MyOctree::getLeafNodes()
{
	if (m_leafNodes.size() > 2)
		return m_leafNodes;
	selectLeafNode(root);
	return m_leafNodes;
}

void MyOctree::cpIntersection(vector<V3d>& edgeIntersections, vector<V3d>& facetIntersections, vector<V3d>& edgeUnitNormals, vector<V3d>& facetUnitNormals)
{
	cout << "We are begining to extract edges" << endl;
	vector<V2i> modelEdges = extractEdges(); 

	cout << "We are begining to select leaves" << endl;
	selectLeafNode(root);

	// 三角形的边与node面交， 因为隐式B样条基定义在了left/bottom/back corner上， 所以与节点只需要求与这三个面的交即可

	for (int i = 0; i < modelEdges.size(); i++)
	{
		Eigen::Vector2i e = modelEdges[i];
		V3d p1 = m_V.row(e.x());
		V3d p2 = m_V.row(e.y());

		double minX = p1.x() < p2.x() ? p1.x() : p2.x();
		double minY = p1.y() < p2.y() ? p1.y() : p2.y();
		double minZ = p1.z() < p2.z() ? p1.z() : p2.z();

		double maxX = p1.x() > p2.x() ? p1.x() : p2.x();
		double maxY = p1.y() > p2.y() ? p1.y() : p2.y();
		double maxZ = p1.z() > p2.z() ? p1.z() : p2.z();

		for (auto node : leafNodes)
		{
			V3d leftBottomBackCorner = node->boundary.first;
			V3d rightTopFrontCorner = node->boundary.second;

			if (maxX < leftBottomBackCorner.x() || minX >= rightTopFrontCorner.x()) continue;
			if (maxY < leftBottomBackCorner.y() || minY >= rightTopFrontCorner.y()) continue;
			if (maxZ < leftBottomBackCorner.z() || minZ >= rightTopFrontCorner.z()) continue;

			if (p1.x() == p2.x())
			{
				// the condition impies that the segment line lies on the plane where the back facet locate on: 
				// or parallel to the back facet and part or the whole of the segment in the node, we don't need to care about the relationship between
				// minX and leftBottomBackCorner.x() because we've considered it before
				if (p1.y() == p2.y())
				{
					// the segment intersect with the bottom facet, we don't need to care about the consitions when 
					// leftBottomBackCorner.z() <= minZ < rightTopFrontCorner.z() and maxZ == rightTopFrontCorner.z() because the extreme point is itself
					if (maxZ > leftBottomBackCorner.z() && minZ < leftBottomBackCorner.z())
					{
						V3d intersection(minX, minY, leftBottomBackCorner.z());
						double a = (p2.z() - intersection.z()) / (p2.z() - p1.z());
						edgeIntersections.emplace_back(intersection);
					}
				}
				else
				{
					// parallel to the bottom facet, i.e., perpendicular to the left facet 
					if (p1.z() == p2.z())
					{
						// the same goes for the above condition
						if (maxY > leftBottomBackCorner.y() && minY < leftBottomBackCorner.y())
						{
							V3d intersection(minX, leftBottomBackCorner.y(), minZ);
							double a = (p2.y() - intersection.y()) / (p2.y() - p1.y());
							edgeIntersections.emplace_back(intersection);
						}
					}
					// the segment is only parallel to the back facet
					else
					{
						// doesn't intersect with the left facet or the intersection is the end point, the conditions can refert to the file "线段与正方形交的几种情况.doc"
						if (minY >= leftBottomBackCorner.y())
						{
							// there doesn't exist intersections
							if (minZ >= leftBottomBackCorner.z()) continue;

							// intersect with the bottom facet or not, the other case the intersection is the end point too
							if (minZ < leftBottomBackCorner.z())
							{
								double z = p2.z() - p1.z();
								double lambda1 = (p2.z() - leftBottomBackCorner.z()) / z;
								double lambda2 = 1 - lambda1;
								double interY = lambda1 * p1.y() + lambda2 * p2.y();
								// intersect with the bottom facet, we don't consider the equal case
								if (interY < rightTopFrontCorner.y())
								{
									edgeIntersections.emplace_back(V3d(p1.x(), interY, leftBottomBackCorner.z()));
								}
							}
						}
						else
						{
							// doesn't intersect with the left facet or the intersection is the end point
							if (minZ >= leftBottomBackCorner.z())
							{
								double y = p2.y() - p1.y();
								double lambda1 = (p2.y() - leftBottomBackCorner.y()) / y;
								double lambda2 = 1 - lambda1;
								double interZ = lambda1 * p1.z() + lambda2 * p2.z();
								// intersect with the left facet, we don't consider the equal case
								if (interZ < rightTopFrontCorner.z())
								{
									edgeIntersections.emplace_back(V3d(p1.x(), leftBottomBackCorner.y(), interZ));
								}
							}
							// the straight line where the segment lies must intersect with the bottom and left facet, we view the corner as two intersections
							// if the corner lies on the line, then what we need to do is to compare the intersections with intervals
							else
							{
								double z = p2.z() - p1.z();
								double lambda1 = (p2.z() - leftBottomBackCorner.z()) / z;
								double lambda2 = 1 - lambda1;
								double interY = lambda1 * p1.y() + lambda2 * p2.y();
								if (interY < rightTopFrontCorner.y() && interY >= leftBottomBackCorner.y())
								{
									edgeIntersections.emplace_back(V3d(p1.x(), interY, leftBottomBackCorner.z()));
								}
								double y = p2.y() - p1.y();
								lambda1 = (p2.y() - leftBottomBackCorner.y()) / y;
								lambda2 = 1 - lambda1;
								double interZ = lambda1 * p1.z() + lambda2 * p2.z();
								if (interZ < rightTopFrontCorner.z() && interZ >= leftBottomBackCorner.z())
								{
									edgeIntersections.emplace_back(V3d(p1.x(), leftBottomBackCorner.y(), interZ));
								}
							}
						}
					}
				}
			}
			else
			{
				// the segment is parallel to the left facet
				if (p1.y() == p2.y())
				{
					if (p1.z() == p2.z())
					{
						// the segment is perpendicular to the back facet 
						if (leftBottomBackCorner.x() > minX && leftBottomBackCorner.x() < maxX)
						{
							V3d intersection(leftBottomBackCorner.x(), p1.y(), p1.z());
							edgeIntersections.emplace_back(intersection);
							double lambda1 = (p2.x() - intersection.x()) / (p2.x() - p1.x());
							double lambda2 = 1 - lambda1;
						}
					}
					// this case is similar to the case when the two end points have the same x value						
					else
					{
						if (minX >= leftBottomBackCorner.x())
						{
							// doesn't intersect with the back facet or the intersection is the end point				
							if (minZ >= leftBottomBackCorner.z()) continue;   // there doesn't exist intersections

							// intersect with the bottom facet or not, the other case the intersection is the end point too
							if (minZ < leftBottomBackCorner.z())
							{
								double z = p2.z() - p1.z();
								double lambda1 = (p2.z() - leftBottomBackCorner.z()) / z;
								double lambda2 = 1 - lambda1;
								double interX = lambda1 * p1.x() + lambda2 * p2.x();
								// intersect with the bottom facet, we don't consider the equal case
								if (interX < rightTopFrontCorner.y() && lambda1 * lambda2 > 0)
								{
									edgeIntersections.emplace_back(V3d(interX, p1.y(), leftBottomBackCorner.z()));
								}
							}
						}
						else
						{
							// doesn't intersect with the back facet or the intersection is the end point
							if (minZ >= leftBottomBackCorner.z())
							{
								double x = p2.x() - p1.x();
								double lambda1 = (p2.x() - leftBottomBackCorner.x()) / x;
								double lambda2 = 1 - lambda1;
								double interZ = lambda1 * p1.z() + lambda2 * p2.z();
								// intersect with the left facet, we don't consider the equal case
								if (interZ < rightTopFrontCorner.z() && lambda1 * lambda2 > 0)
								{
									edgeIntersections.emplace_back(V3d(leftBottomBackCorner.x(), p1.y(), interZ));
								}
							}
							// the straight line where the segment lies must intersect with the bottom and back facet, we view the corner(may be not the node's corner
							//  as two intersections if the corner lies on the line, then what we need to do is to compare the intersections with intervals
							else
							{
								double z = p2.z() - p1.z();
								double lambda1 = (p2.z() - leftBottomBackCorner.z()) / z;
								double lambda2 = 1 - lambda1;
								if (lambda1 * lambda2 > 0)
								{
									double interX = lambda1 * p1.x() + lambda2 * p2.x();
									if (interX < rightTopFrontCorner.x() && interX >= leftBottomBackCorner.x())
									{
										edgeIntersections.emplace_back(V3d(interX, p1.y(), leftBottomBackCorner.z()));
									}
								}
								double x = p2.x() - p1.x();
								lambda1 = (p2.x() - leftBottomBackCorner.x()) / x;
								lambda2 = 1 - lambda1;
								if (lambda1 * lambda2 > 0)
								{
									double interZ = lambda1 * p1.z() + lambda2 * p2.z();
									if (interZ < rightTopFrontCorner.z() && interZ >= leftBottomBackCorner.z())
									{
										edgeIntersections.emplace_back(V3d(leftBottomBackCorner.x(), p1.y(), interZ));
									}
								}

							}
						}
					}
				}
				// the straight line where the segment lies must intersect with the left, bottom and back facet, we view the corner(may be not the node's corner
				//  as three intersections if the corner lies on the line, then what we need to do is to compare the intersections with intervals
				else
				{
					// intersect with the bottom facet
					double z = p2.z() - p1.z();
					double lambda1 = (p2.z() - leftBottomBackCorner.z()) / z;
					double lambda2 = 1 - lambda1;
					if (lambda1 * lambda2 > 0)
					{
						double interX = lambda1 * p1.x() + lambda2 * p2.x();
						double interY = lambda1 * p1.y() + lambda2 * p2.y();
						if (interX < rightTopFrontCorner.x() && interX >= leftBottomBackCorner.x() && interY < rightTopFrontCorner.y() && interY >= leftBottomBackCorner.y())
						{
							edgeIntersections.emplace_back(V3d(interX, interY, leftBottomBackCorner.z()));
						}
					}
					// intersect with the back facet
					double x = p2.x() - p1.x();
					lambda1 = (p2.x() - leftBottomBackCorner.x()) / x;
					lambda2 = 1 - lambda1;
					if (lambda1 * lambda2 > 0)
					{
						double interY = lambda1 * p1.y() + lambda2 * p2.y();
						double interZ = lambda1 * p1.z() + lambda2 * p2.z();
						if (interY < rightTopFrontCorner.y() && interY >= leftBottomBackCorner.y() && interZ < rightTopFrontCorner.z() && interZ >= leftBottomBackCorner.z())
						{
							edgeIntersections.emplace_back(V3d(leftBottomBackCorner.x(), interY, interZ));
						}
					}
					// intersect with the left facet
					double y = p2.y() - p1.y();
					lambda1 = (p2.y() - leftBottomBackCorner.y()) / y;
					lambda2 = 1 - lambda1;
					if (lambda1 * lambda2 > 0)
					{
						double interX = lambda1 * p1.x() + lambda2 * p2.x();
						double interZ = lambda1 * p1.z() + lambda2 * p2.z();
						if (interX < rightTopFrontCorner.x() && interX >= leftBottomBackCorner.x() && interZ < rightTopFrontCorner.z() && interZ >= leftBottomBackCorner.z())
						{
							edgeIntersections.emplace_back(V3d(interX, leftBottomBackCorner.y(), interZ));
						}
					}
				}
			}
		}
		std::cout << "\r Computing[" << fixed << setprecision(2) << i * outpute << "%]";
		//int show_num = i / 50;
		/*for (int j = 1; j <= show_num; j++)
		{
			cout << "#";
			Sleep(10);
		}*/
	}
	std::cout << "#####################################################################." << endl;

	// 三角形面与node边线交
	std::cout << "Continue to compute the intersections for triangle faces..." << endl;
	for (int i = 0; i < m_faces.size(); i++)
	{
		std::cout << "\r Computing[" << std::fixed << std::setprecision(2) << i * outputf << "%]";

		V3i f = m_faces[i];
		V3d p1 = m_normalizedVerts[f.x()];
		V3d p2 = m_normalizedVerts[f.y()];
		V3d p3 = m_normalizedVerts[f.z()];
		Eigen::Matrix3d faceMatrix;
		faceMatrix << p1.x(), p2.x(), p3.x(),
			p1.y(), p2.y(), p3.y(),
			p1.z(), p2.z(), p3.z();
		V3d maxElement = faceMatrix.rowwise().maxCoeff();
		V3d minElement = faceMatrix.rowwise().minCoeff();

		Triangle t(p1, p2, p3);

		for (pair<V3d, V3d> edge : nodeXEdges[depth])
		{
			double y = edge.first.y();
			double z = edge.first.z();
			if (maxElement.x() <= edge.first.x() || minElement.x() >= edge.second.x()) continue;
			if (minElement.y() >= y || maxElement.y() <= y || minElement.z() >= z || maxElement.z() <= z) continue;
			V3d coefficient = t.ComputeCoefficientOfTriangle(y, z, 1);
			if (coefficient.x() > 0)
			{
				double interX = coefficient.x() * p1.x() + coefficient.y() * p2.x() + coefficient.z() * p3.x();
				if (interX >= edge.first.x() && interX <= edge.second.x())
				{
					facetIntersections.emplace_back(V3d(interX, y, z));
					facetUnitNormals.emplace_back(m_FaceNormals[i]);
				}
			}
		}

		for (pair<V3d, V3d> edge : nodeYEdges[depth])
		{
			double x = edge.first.x();
			double z = edge.first.z();
			if (maxElement.y() <= edge.first.y() || minElement.y() >= edge.second.y()) continue;
			if (minElement.x() >= x || maxElement.x() <= x || minElement.z() >= z || maxElement.z() <= z) continue;
			V3d coefficient = t.ComputeCoefficientOfTriangle(z, x, 2);
			if (coefficient.x() > 0)
			{
				double interY = coefficient.x() * p1.y() + coefficient.y() * p2.y() + coefficient.z() * p3.y();
				if (interY >= edge.first.y() && interY <= edge.second.y())
				{
					facetIntersections.emplace_back(V3d(x, interY, z));
					facetUnitNormals.emplace_back(m_FaceNormals[i]);
				}
			}
		}

		for (pair<V3d, V3d> edge : nodeZEdges[depth])
		{
			double x = edge.first.x();
			double y = edge.first.y();
			if (maxElement.z() <= edge.first.z() || minElement.z() >= edge.second.z()) continue;
			if (minElement.x() >= x || maxElement.x() <= x || minElement.y() >= y || maxElement.y() <= y) continue;
			V3d coefficient = t.ComputeCoefficientOfTriangle(x, y, 0);
			if (coefficient.x() > 0)
			{
				double interZ = coefficient.x() * p1.z() + coefficient.y() * p2.z() + coefficient.z() * p3.z();
				if (interZ >= edge.first.z() && interZ <= edge.second.z())
				{
					facetIntersections.emplace_back(V3d(x, y, interZ));
					facetUnitNormals.emplace_back(m_FaceNormals[i]);
				}
			}
		}
	}
	std::cout << "#####################################################################." << endl;
}

void  MyOctree::SaveIntersections(string filename, vector<V3d> intersections, vector<V3d>& UnitNormals) const
{
	ofstream out(filename);
	for (V3d p : intersections)
	{
		out << "v " << p.x() << " " << p.y() << " " << p.z() << endl;
	}
	for (V3d p : UnitNormals)
	{
		out << "vn " << p.x() << " " << p.y() << " " << p.z() << endl;
	}
	out.close();
}

SpMat MyOctree::CeofficientOfPoints(const vector<V3d> edgeIntersections, const vector<V3d> faceIntersections, SpMat& Bx, SpMat& By, SpMat& Bz)
{
	vector<V3d> allPoints;
	for (V3d p : m_normalizedVerts) allPoints.emplace_back(p);
	for (V3d p : edgeIntersections) allPoints.emplace_back(p);
	for (V3d p : faceIntersections) allPoints.emplace_back(p);
	int N_points = allPoints.size();
	int N_leafNodes = m_leafNodes.size();

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
			V3d nodeCorner = m_leafNodes[j]->boundary.first;
			double width = m_leafNodes[j]->width;
			double x = BaseFunction(allPoints[i].x(), width, nodeCorner.x());
			double y = BaseFunction(allPoints[i].y(), width, nodeCorner.y());
			double z = BaseFunction(allPoints[i].z(), width, nodeCorner.z());


			double dx = dBaseFunction(allPoints[i].x(), width, nodeCorner.x());
			double dy = dBaseFunction(allPoints[i].y(), width, nodeCorner.y());
			double dz = dBaseFunction(allPoints[i].z(), width, nodeCorner.z());
			double w3 = width * width * width;
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

Eigen::VectorXd MyOctree::Solver(Eigen::VectorXd& VertexValue, const double alpha)
{
	Eigen::VectorXd X;
	vector<V3d> edgeIntersections;
	vector<V3d> facetIntersections;
	vector<V3d> edgeUnitNormals;
	vector<V3d> facetUnitNormals;
	Intersection(edgeIntersections, facetIntersections, edgeUnitNormals, facetUnitNormals);
	SaveIntersections("data\\" + model_Name + "\\" + to_string(m_maxDepth) + "edge_intersections.obj", edgeIntersections, edgeUnitNormals);
	SaveIntersections("data\\" + model_Name + "\\" + to_string(m_maxDepth) + "facet_intersections.obj", facetIntersections, facetUnitNormals);
	vector<V3d> normals;
	for (V3d n : m_VertNormals) normals.emplace_back(n);
	for (V3d n : edgeUnitNormals) normals.emplace_back(n);
	for (V3d n : facetUnitNormals) normals.emplace_back(n);
	SpMat Bx;
	SpMat By;
	SpMat Bz;
	SpMat B = CeofficientOfPoints(edgeIntersections, facetIntersections, Bx, By, Bz);
	//cout << B.row(0) << endl;
	//cout << Bx.row(0) << endl;
	//cout << By.row(0) << endl;
	//cout << Bz.row(0) << endl;

	SpMat A = alpha * B.transpose() * B + Bx.transpose() * Bx + By.transpose() * By + Bz.transpose() * Bz;
	A.makeCompressed();

	Eigen::MatrixXd normalMatrix(normals.size(), 3);
	for (int i = 0; i < normals.size(); i++)
	{
		V3d n = normals[i];
		for (int j = 0; j < 3; j++) normalMatrix(i, j) = n(j);

	}
	Eigen::VectorXd b;
	b = Bx.transpose() * normalMatrix.col(0) + By.transpose() * normalMatrix.col(1) + Bz.transpose() * normalMatrix.col(2);
	Eigen::LeastSquaresConjugateGradient<SpMat> solver_sparse;

	solver_sparse.setTolerance(1e-6);
	solver_sparse.compute(A);
	X = solver_sparse.solve(b);
	SaveBValue2TXT("data\\" + model_Name + "\\" + to_string(m_maxDepth) + "BSplineCoefficient.txt", X);
	string filename = "data\\" + model_Name + "\\" + to_string(m_maxDepth) + "BSplineDataAtVertices.txt";

	VertexValue = B * X;
	SaveBValue2TXT(filename, VertexValue);
	cout << B * X - b << endl;
	return X;
}

void MyOctree::SaveBValue2TXT(string filename, Eigen::VectorXd X) const
{
	ofstream out(filename);
	out << setiosflags(ios::fixed) << setprecision(9) << X << endl;
	out.close();
}
