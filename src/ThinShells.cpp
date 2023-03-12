#include "ThinShells.h"

MarchingCubes::MarchingCubes(const VXd& value, const MXd& nodes, const double& isoValue, const int& resolution,
	vector<V3d>& MV, vector<V3i>& MF)
{
	m_value = value;
	m_nodes = nodes;
	m_isoValue = isoValue;
	m_resolution = resolution;

	ExtractIsoSurface(MV, MF);
}

V3d MarchingCubes::Interpolate(V3d p1, V3d p2, double v1, double v2, double iso)
{
	float mu;
	V3d newPoint;
	if (abs(iso - v1) < 1e-10)
		return p1;
	if (abs(iso - v2) < 1e-10)
		return p2;
	if (abs(v1 - v2) < 1e-10)
		return p1;

	mu = (iso - v1) / (v2 - v1);
	newPoint.x() = p1.x() + mu * (p2.x() - p1.x());
	newPoint.y() = p1.y() + mu * (p2.y() - p1.y());
	newPoint.z() = p1.z() + mu * (p2.z() - p1.z());
	return newPoint;
}

void MarchingCubes::ExtractIsoSurface(vector<V3d>& Vertices, vector<V3i>& Faces)
{
	Vertices.clear();
	Faces.clear();
	int VertexId = 0;
	int nxtResolution = m_resolution + 1;
	int square = nxtResolution * nxtResolution;
	for (int i = 0; i < m_resolution; i++)
	{
		for (int j = 0; j < m_resolution; j++)
		{
			for (int k = 0; k < m_resolution; k++)
			{
				vector<int> nodesID;
				int cubeCase = 0;
				int idx = 1;
				for (int c : {0, 1})
				{
					for (int b : {0, 1})
					{
						if (b == 0)
						{
							for (int a : {0, 1})
							{
								nodesID.emplace_back(c + k + j * nxtResolution + (i + a) * square);
							}
						}
						else
						{
							for (int a : {1, 0})
							{
								nodesID.emplace_back(c + k + (1 + j) * nxtResolution + (i + a) * square);
							}
						}
					}
				}
				for (int kk : nodesID)
				{
					if (m_value[kk] > m_isoValue) cubeCase |= idx;
					//cout << node->SDFValue[i] << " ";
					idx *= 2;
				}

				if (cubeCases[cubeCase] == 0)
					continue;

				V3d vertexList[12];
				//Find the vertices where the surface intersects the cube
				if (cubeCases[cubeCase] & 1)
				{
					if ((m_value[nodesID[0]] > m_isoValue) != (m_value[nodesID[1]] > m_isoValue))
						vertexList[0] = Interpolate(m_nodes.row(nodesID[0]), m_nodes.row(nodesID[1]), m_value[nodesID[0]], m_value[nodesID[1]], m_isoValue);
				}

				if (cubeCases[cubeCase] & 2)
				{
					if ((m_value[nodesID[1]] > m_isoValue) != (m_value[nodesID[2]] > m_isoValue))
						vertexList[1] = Interpolate(m_nodes.row(nodesID[1]), m_nodes.row(nodesID[2]), m_value[nodesID[1]], m_value[nodesID[2]], m_isoValue);
				}
				if (cubeCases[cubeCase] & 4)
				{
					if ((m_value[nodesID[2]] > m_isoValue) != (m_value[nodesID[3]] > m_isoValue))
						vertexList[2] = Interpolate(m_nodes.row(nodesID[2]), m_nodes.row(nodesID[3]), m_value[nodesID[2]], m_value[nodesID[3]], m_isoValue);
				}
				if (cubeCases[cubeCase] & 8)
				{
					if ((m_value[nodesID[3]] > m_isoValue) != (m_value[nodesID[0]] > m_isoValue))
						vertexList[3] = Interpolate(m_nodes.row(nodesID[3]), m_nodes.row(nodesID[0]), m_value[nodesID[3]], m_value[nodesID[0]], m_isoValue);
				}
				if (cubeCases[cubeCase] & 16)
				{
					if ((m_value[nodesID[4]] > m_isoValue) != (m_value[nodesID[5]] > m_isoValue))
						vertexList[4] = Interpolate(m_nodes.row(nodesID[4]), m_nodes.row(nodesID[5]), m_value[nodesID[4]], m_value[nodesID[5]], m_isoValue);
				}
				if (cubeCases[cubeCase] & 32)
				{
					if ((m_value[nodesID[5]] > m_isoValue) != (m_value[nodesID[6]] > m_isoValue))
						vertexList[5] = Interpolate(m_nodes.row(nodesID[5]), m_nodes.row(nodesID[6]), m_value[nodesID[5]], m_value[nodesID[6]], m_isoValue);
				}
				if (cubeCases[cubeCase] & 64)
				{
					if ((m_value[nodesID[6]] > m_isoValue) != (m_value[nodesID[7]] > m_isoValue))
						vertexList[6] = Interpolate(m_nodes.row(nodesID[6]), m_nodes.row(nodesID[7]), m_value[nodesID[6]], m_value[nodesID[7]], m_isoValue);
				}
				if (cubeCases[cubeCase] & 128)
				{
					if ((m_value[nodesID[7]] > m_isoValue) != (m_value[nodesID[4]] > m_isoValue))
						vertexList[7] = Interpolate(m_nodes.row(nodesID[7]), m_nodes.row(nodesID[4]), m_value[nodesID[7]], m_value[nodesID[4]], m_isoValue);
				}
				if (cubeCases[cubeCase] & 256)
				{
					if ((m_value[nodesID[0]] > m_isoValue) != (m_value[nodesID[4]] > m_isoValue))
						vertexList[8] = Interpolate(m_nodes.row(nodesID[0]), m_nodes.row(nodesID[4]), m_value[nodesID[0]], m_value[nodesID[4]], m_isoValue);
				}
				if (cubeCases[cubeCase] & 512)
				{
					if ((m_value[nodesID[1]] > m_isoValue) != (m_value[nodesID[5]] > m_isoValue))
						vertexList[9] = Interpolate(m_nodes.row(nodesID[1]), m_nodes.row(nodesID[5]), m_value[nodesID[1]], m_value[nodesID[5]], m_isoValue);
				}
				if (cubeCases[cubeCase] & 1024)
				{
					if ((m_value[nodesID[2]] > m_isoValue) != (m_value[nodesID[6]] > m_isoValue))
						vertexList[10] = Interpolate(m_nodes.row(nodesID[2]), m_nodes.row(nodesID[6]), m_value[nodesID[2]], m_value[nodesID[6]], m_isoValue);
				}
				if (cubeCases[cubeCase] & 2048)
				{
					if ((m_value[nodesID[3]] > m_isoValue) != (m_value[nodesID[7]] > m_isoValue))
						vertexList[11] = Interpolate(m_nodes.row(nodesID[3]), m_nodes.row(nodesID[7]), m_value[nodesID[3]], m_value[nodesID[7]], m_isoValue);
				}

				for (int ii = 0; triangleTable[cubeCase][ii] != -1; ii = ii + 3)
				{
					Vertices.emplace_back(vertexList[triangleTable[cubeCase][ii]]);
					Vertices.emplace_back(vertexList[triangleTable[cubeCase][ii + 1]]);
					Vertices.emplace_back(vertexList[triangleTable[cubeCase][ii + 2]]);
					Faces.emplace_back(V3i(VertexId, VertexId + 1, VertexId + 2));
					VertexId += 3;
				}
			}
		}
	}
	
}

ThinShells::ThinShells(const int& resolution, const string& modelName, const string& fileFormat)
{
	m_resolution = resolution;
	m_modelName = modelName;
	readFile(SHARED_PATH + modelName + fileFormat);
	ConstructUniformNode(m_resolution, modelName);
	SaveNode(OUT_PATH + modelName + "/" + std::to_string(m_resolution) + "node.obj");
	vector<V3d> intersections;
	vector<vector<int>> relativeNodes;
	Intersection(intersections, relativeNodes);
	saveVertices(OUT_PATH + modelName + "/" + std::to_string(m_resolution) + "intersections.obj", intersections);
	VXd BSplineCoefficient = ImplicitField(intersections, relativeNodes);
	
}

ThinShells& ThinShells::operator=(const ThinShells& model)
{
	this->m_width = model.m_width;
	this->m_nodes = model.m_nodes;
	this->m_BSplineCoefficient = model.m_BSplineCoefficient;
	this->m_resolution = model.m_resolution;
	this->m_shellValue = model.m_shellValue;
	this->m_modelName = model.m_modelName;
	return *this;
}

void ThinShells::ConstructUniformNode(const int& resolution, const string& modelName)
{
	//m_resolution = resolution;
	/*m_modelName = modelName;*/
	m_V = Eigen::Map<MXd>(modelVerts.data()->data(), modelVerts.size(), 3);
	m_F = Eigen::Map<MXi>(modelFaces.data()->data(), modelFaces.size(), 3);

	V3d maxV = m_V.colwise().maxCoeff();
	V3d minV = m_V.colwise().minCoeff();
	// 保证外围格子是空的
	V3d beginV = minV - (maxV - minV) * 0.1;
	V3d endV = maxV + (maxV - minV) * 0.1;
	V3d width = (endV - beginV) / m_resolution;
	m_width = width;
	/*cout << "max = " << maxV.transpose() << "   min = " << minV.transpose() << endl;
	cout << "max = " << endV.transpose() << "   min = " << beginV.transpose() << endl;*/

	vector<V3d> Node;
	for (int i = 0; i <= m_resolution; i++)
	{
		double pX = i * width.x() + beginV.x();
		for (int j = 0; j <= m_resolution; j++)
		{
			double pY = j * width.y() + beginV.y();
			for (int k = 0; k <= m_resolution; k++)
			{
				double pZ = k * width.z() + beginV.z();
				Node.emplace_back(V3d(pX, pY, pZ));
			}
		}
	}
	m_nodes = Eigen::Map<MXd>(Node.data()->data(), Node.size(), 3);
	Node.clear();
}

void ThinShells::SaveNode(const string& filename) const
{
	std::ofstream out(filename);
	int count = -8;
	for (int i = 0; i < m_nodes.rows(); i++)
	{
		V3d corner = m_nodes.row(i);
		vector<V3d> corners;
		corners.resize(8);
		double minX = corner.x();
		double minY = corner.y();
		double minZ = corner.z();
		double maxX = corner.x() + m_width.x();
		double maxY = corner.y() + m_width.y();
		double maxZ = corner.z() + m_width.z();

		corners[0] = corner;
		corners[1] = V3d(maxX, minY, minZ);
		corners[2] = V3d(minX, maxY, minZ);
		corners[3] = V3d(maxX, maxY, minZ);
		corners[4] = V3d(minX, minY, maxZ);
		corners[5] = V3d(maxX, minY, maxZ);
		corners[6] = V3d(minX, maxY, maxZ);
		corners[7] = V3d(maxX, maxY, maxZ);

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

V3d ThinShells::ComputeCoefficientOfTriangle(const V3d& p1, const V3d& p2, const V3d& p3, const double& x, const double& y, int flag)
{
	V3d d1 = p2 - p1;
	V3d d2 = p3 - p1;
	if (flag == 0)
	{
		V3d intersection = V3d(x, y, 0.0);
		double area = d1.cross(d2).z();
		if (abs(area) < 1e-10) return V3d(-1.0, -1.0, -1.0);
		V3d v1 = p1 - intersection;
		V3d v2 = p2 - intersection;
		V3d v3 = p3 - intersection;

		double area1 = v2.cross(v3).z();
		double lambda1 = area1 / area;
		if (lambda1 < 0) return V3d(-1.0, -1.0, -1.0);
		area1 = v3.cross(v1).z();
		double lambda2 = area1 / area;
		if (lambda2 < 0) return V3d(-1.0, -1.0, -1.0);
		area1 = v1.cross(v2).z();
		double lambda3 = area1 / area;
		if (lambda3 < 0) return V3d(-1.0, -1.0, -1.0);
		return V3d(lambda1, lambda2, lambda3);
	}
	if (flag == 1)
	{
		V3d intersection(0.0, x, y);
		double area = d1.cross(d2).x();
		if (abs(area) < 1e-10) return V3d(-1.0, -1.0, -1.0);
		V3d v1 = p1 - intersection;
		V3d v2 = p2 - intersection;
		V3d v3 = p3 - intersection;

		double area1 = v2.cross(v3).x();
		double lambda1 = area1 / area;
		if (lambda1 < 0) return V3d(-1.0, -1.0, -1.0);
		area1 = v3.cross(v1).x();
		double lambda2 = area1 / area;
		if (lambda2 < 0) return V3d(-1.0, -1.0, -1.0);
		area1 = v1.cross(v2).x();
		double lambda3 = area1 / area;
		if (lambda3 < 0) return V3d(-1.0, -1.0, -1.0);
		return V3d(lambda1, lambda2, lambda3);
	}

	V3d intersection = V3d(y, 0.0, x);
	double area = d1.cross(d2).y();
	if (abs(area) < 1e-10) return V3d(-1.0, -1.0, -1.0);
	V3d v1 = p1 - intersection;
	V3d v2 = p2 - intersection;
	V3d v3 = p3 - intersection;

	double area1 = v2.cross(v3).y();
	double lambda1 = area1 / area;
	if (lambda1 < 0) return V3d(-1.0, -1.0, -1.0);
	area1 = v3.cross(v1).y();
	double lambda2 = area1 / area;
	if (lambda2 < 0) return V3d(-1.0, -1.0, -1.0);
	area1 = v1.cross(v2).y();
	double lambda3 = area1 / area;
	if (lambda3 < 0) return V3d(-1.0, -1.0, -1.0);
	return V3d(lambda1, lambda2, lambda3);
}

void ThinShells::Intersection(vector<V3d>& Intersections, vector<vector<int>>& relativeNodes)
{
	int nodeNumber = m_nodes.rows();
	int square = (m_resolution + 1) * (m_resolution + 1);
	cout << "nodeNumber: " << nodeNumber << endl;
	double outputf = 100.0 / m_F.rows();    // used for output
	// 三角形面与node边线交
	cout << "computing the intersections for triangle faces..." << endl;
	for (int ii = 0; ii < m_F.rows(); ii++)
	{
		std::cout << "\r Computing[" << std::fixed << std::setprecision(2) << ii * outputf << "%]";

		V3i f = m_F.row(ii);
		V3d p1 = m_V.row(f.x());
		V3d p2 = m_V.row(f.y());
		V3d p3 = m_V.row(f.z());
		Eigen::Matrix3d faceMatrix;
		faceMatrix << p1.x(), p2.x(), p3.x(),
			p1.y(), p2.y(), p3.y(),
			p1.z(), p2.z(), p3.z();
		V3d maxElement = faceMatrix.rowwise().maxCoeff();
		V3d minElement = faceMatrix.rowwise().minCoeff();

		for (int i = 0; i < nodeNumber; i++)
		{
			double y = m_nodes.row(i).y();
			if (maxElement.y() <= y || minElement.y() >= y) continue;
			double z = m_nodes.row(i).z();
			if (maxElement.z() <= z || minElement.z() >= z) continue;

			V3d coefficient = ComputeCoefficientOfTriangle(p1, p2, p3, y, z, 1);
			if (coefficient.x() > 0)
			{
				double interX = coefficient.x() * p1.x() + coefficient.y() * p2.x() + coefficient.z() * p3.x();
				if (interX >= m_nodes.row(i).x() && interX <= m_nodes.row(i).x() + m_width.x())
				{
					Intersections.emplace_back(V3d(interX, y, z));
					vector<int> nodes(2);
					nodes[0] = i;
					nodes[1] = i + square;
					relativeNodes.emplace_back(nodes);
					/*cout << i << " " << m_nodes.row(i) << "        " << interX << "    width = " << m_width.x() << "   " << 
						nodes[1] << "  " << m_nodes.row(nodes[1]) << endl;*/
				}
			}
		}

		for (int i = 0; i < nodeNumber; i++)
		{
			double x = m_nodes.row(i).x();
			if (maxElement.x() <= x || minElement.x() >= x) continue;
			double z = m_nodes.row(i).z();
			if (maxElement.z() <= z || minElement.z() >= z) continue;

			V3d coefficient = ComputeCoefficientOfTriangle(p1, p2, p3, z, x, 2);
			if (coefficient.x() > 0)
			{
				double interY = coefficient.x() * p1.y() + coefficient.y() * p2.y() + coefficient.z() * p3.y();
				if (interY >= m_nodes.row(i).y() && interY <= m_nodes.row(i).y() + m_width.y())
				{
					Intersections.emplace_back(V3d(x, interY, z));
					vector<int> nodes(2);
					nodes[0] = i;
					nodes[1] = i + m_resolution + 1;
					relativeNodes.emplace_back(nodes);
					/*cout << i << " " << m_nodes.row(i) << "        " << interY << "    width = " << m_width.y() << "   " <<
						nodes[1] << "  " << m_nodes.row(nodes[1]) << endl;*/
				}
			}
		}

		for (int i = 0; i < nodeNumber; i++)
		{
			double x = m_nodes.row(i).x();
			if (maxElement.x() <= x || minElement.x() >= x) continue;
			double y = m_nodes.row(i).y();
			if (maxElement.y() <= y || minElement.y() >= y) continue;
			V3d coefficient = ComputeCoefficientOfTriangle(p1, p2, p3, x, y, 0);
			if (coefficient.x() > 0)
			{
				double interZ = coefficient.x() * p1.z() + coefficient.y() * p2.z() + coefficient.z() * p3.z();
				if (interZ >= m_nodes.row(i).z() && interZ <= m_nodes.row(i).z() + m_width.z())
				{
					Intersections.emplace_back(V3d(x, y, interZ));
					vector<int> nodes(2);
					nodes[0] = i;
					nodes[1] = i + 1;
					relativeNodes.emplace_back(nodes);
					/*cout << i << " " << m_nodes.row(i) << "        " << interZ << "    width = " << m_width.x() << "   " <<
						nodes[1] << "  " << m_nodes.row(nodes[1]) << endl;*/
				}
			}
		}
	}
	cout << "#####################################################################." << endl;

	// **************************************************************************************************************
	// 三角形的边与node面交， 因为隐式B样条基定义在了left/bottom/back corner上， 所以与节点只需要求与这三个面的交即可
	std::cout << "Continue to compute the intersections for triangle edges..." << std::endl;
	//extract the edges of the input triangle mesh(V, F)
	std::vector<V2i> edges = extractEdges();
	double outpute = 100.0 / edges.size();
	for (int ii = 0; ii < edges.size(); ii++)
	{
		V2i e = edges[ii];
		V3d p1 = m_V.row(e.x());
		V3d p2 = m_V.row(e.y());

		double minX = p1.x() < p2.x() ? p1.x() : p2.x();
		double minY = p1.y() < p2.y() ? p1.y() : p2.y();
		double minZ = p1.z() < p2.z() ? p1.z() : p2.z();

		double maxX = p1.x() > p2.x() ? p1.x() : p2.x();
		double maxY = p1.y() > p2.y() ? p1.y() : p2.y();
		double maxZ = p1.z() > p2.z() ? p1.z() : p2.z();

		for (int i = 0; i < nodeNumber; i++)
		{
			V3d leftBottomBackCorner = m_nodes.row(i);
			V3d rightTopFrontCorner = leftBottomBackCorner + m_width;
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
						Intersections.emplace_back(intersection);
						vector<int> nodes(4);
						nodes[0] = i;
						nodes[1] = i + square;
						nodes[2] = i + m_resolution + 1;
						nodes[3] = nodes[1] + m_resolution + 1;
						relativeNodes.emplace_back(nodes);
						/*for (int lk = 0; lk < 4; lk++)
							cout << m_nodes.row(nodes[lk]) << endl;
						cout << intersection << endl;
						cout << "==================================================" << endl;*/
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
							Intersections.emplace_back(intersection);
							vector<int> nodes(4);
							nodes[0] = i;
							nodes[1] = i + square;
							nodes[2] = i + 1;
							nodes[3] = nodes[1] + 1;
							relativeNodes.emplace_back(nodes);

							/*for (int lk = 0; lk < 4; lk++)
								cout << m_nodes.row(nodes[lk]) << endl;
							cout << intersection << endl;
							cout << "==================================================" << endl;*/
						}
					}
					// the segment is only parallel to the back facet
					else
					{
						// doesn't intersect with the left facet or the intersection is the end point, the conditions can refert to the file 
						// "线段与正方形交的几种情况.doc"
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
									Intersections.emplace_back(V3d(p1.x(), interY, leftBottomBackCorner.z()));
									vector<int> nodes(4);
									nodes[0] = i;
									nodes[1] = i + square;
									nodes[2] = i + m_resolution + 1;
									nodes[3] = nodes[1] + m_resolution + 1;
									relativeNodes.emplace_back(nodes);

								/*	for (int lk = 0; lk < 4; lk++)
										cout << m_nodes.row(nodes[lk]) << endl;
									cout << V3d(p1.x(), interY, leftBottomBackCorner.z()) << endl;
									cout << "==================================================" << endl;*/
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
									Intersections.emplace_back(V3d(p1.x(), leftBottomBackCorner.y(), interZ));
									vector<int> nodes(4);
									nodes[0] = i;
									nodes[1] = i + square;
									nodes[2] = i + 1;
									nodes[3] = nodes[1] + 1;
									relativeNodes.emplace_back(nodes);

									/*for (int lk = 0; lk < 4; lk++)
										cout << m_nodes.row(nodes[lk]) << endl;
									cout << V3d(p1.x(), leftBottomBackCorner.y(), interZ) << endl;
									cout << "==================================================" << endl;*/
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
									Intersections.emplace_back(V3d(p1.x(), interY, leftBottomBackCorner.z()));
									vector<int> nodes(4);
									nodes[0] = i;
									nodes[1] = i + square;
									nodes[2] = i + m_resolution + 1;
									nodes[3] = nodes[1] + m_resolution + 1;
									relativeNodes.emplace_back(nodes);

									/*for (int lk = 0; lk < 4; lk++)
										cout << m_nodes.row(nodes[lk]) << endl;
									cout << V3d(p1.x(), interY, leftBottomBackCorner.z()) << endl;
									cout << "==================================================" << endl;*/
								}
								double y = p2.y() - p1.y();
								lambda1 = (p2.y() - leftBottomBackCorner.y()) / y;
								lambda2 = 1 - lambda1;
								double interZ = lambda1 * p1.z() + lambda2 * p2.z();
								if (interZ < rightTopFrontCorner.z() && interZ >= leftBottomBackCorner.z())
								{
									Intersections.emplace_back(V3d(p1.x(), leftBottomBackCorner.y(), interZ));
									vector<int> nodes(4);
									nodes[0] = i;
									nodes[1] = i + square;
									nodes[2] = i + 1;
									nodes[3] = nodes[1] + 1;
									relativeNodes.emplace_back(nodes);

									/*for (int lk = 0; lk < 4; lk++)
										cout << m_nodes.row(nodes[lk]) << endl;
									cout << V3d(p1.x(), leftBottomBackCorner.y(), interZ) << endl;
									cout << "==================================================" << endl;*/
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
							Intersections.emplace_back(intersection);
							vector<int> nodes(4);
							nodes[0] = i;
							nodes[1] = i + m_resolution + 1;
							nodes[2] = i + 1;
							nodes[3] = nodes[1] + 1;
							relativeNodes.emplace_back(nodes);

							/*for (int lk = 0; lk < 4; lk++)
								cout << m_nodes.row(nodes[lk]) << endl;
							cout << intersection << endl;
							cout << "==================================================" << endl;*/
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
									Intersections.emplace_back(V3d(interX, p1.y(), leftBottomBackCorner.z()));
									vector<int> nodes(4);
									nodes[0] = i;
									nodes[1] = i + square;
									nodes[2] = i + m_resolution + 1;
									nodes[3] = nodes[1] + m_resolution + 1;
									relativeNodes.emplace_back(nodes);

									/*for (int lk = 0; lk < 4; lk++)
										cout << m_nodes.row(nodes[lk]) << endl;
									cout << V3d(interX, p1.y(), leftBottomBackCorner.z()) << endl;
									cout << "==================================================" << endl;*/
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
									Intersections.emplace_back(V3d(leftBottomBackCorner.x(), p1.y(), interZ));
									vector<int> nodes(4);
									nodes[0] = i;
									nodes[1] = i + m_resolution + 1;
									nodes[2] = i + 1;
									nodes[3] = nodes[1] + 1;
									relativeNodes.emplace_back(nodes);

									/*for (int lk = 0; lk < 4; lk++)
										cout << m_nodes.row(nodes[lk]) << endl;
									cout << V3d(leftBottomBackCorner.x(), p1.y(), interZ) << endl;
									cout << "==================================================" << endl;*/
								}
							}
							// the straight line where the segment lies must intersect with the bottom and back facet, we view the corner(may be not
							// the node's corner as two intersections if the corner lies on the line, then what we need to do is to compare the
							// intersections with intervals
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
										Intersections.emplace_back(V3d(interX, p1.y(), leftBottomBackCorner.z()));
										vector<int> nodes(4);
										nodes[0] = i;
										nodes[1] = i + square;
										nodes[2] = i + m_resolution + 1;
										nodes[3] = nodes[1] + m_resolution + 1;
										relativeNodes.emplace_back(nodes);

										/*for (int lk = 0; lk < 4; lk++)
											cout << m_nodes.row(nodes[lk]) << endl;
										cout << V3d(interX, p1.y(), leftBottomBackCorner.z()) << endl;
										cout << "==================================================" << endl;*/
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
										Intersections.emplace_back(V3d(leftBottomBackCorner.x(), p1.y(), interZ));
										vector<int> nodes(4);
										nodes[0] = i;
										nodes[1] = i + m_resolution + 1;
										nodes[2] = i + 1;
										nodes[3] = nodes[1] + 1;
										relativeNodes.emplace_back(nodes);

										/*for (int lk = 0; lk < 4; lk++)
											cout << m_nodes.row(nodes[lk]) << endl;
										cout << V3d(leftBottomBackCorner.x(), p1.y(), interZ) << endl;
										cout << "==================================================" << endl;*/
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
						if (interX < rightTopFrontCorner.x() && interX >= leftBottomBackCorner.x() && interY < rightTopFrontCorner.y() &&
							interY >= leftBottomBackCorner.y())
						{
							Intersections.emplace_back(V3d(interX, interY, leftBottomBackCorner.z()));
							vector<int> nodes(4);
							nodes[0] = i;
							nodes[1] = i + square;
							nodes[2] = i + m_resolution + 1;
							nodes[3] = nodes[1] + m_resolution + 1;
							relativeNodes.emplace_back(nodes);
							/*for (int lk = 0; lk < 4; lk++)
								cout << nodes[lk] << "        " << m_nodes.row(nodes[lk]) << endl;
							cout << V3d(interX, interY, leftBottomBackCorner.z()) << endl;
							cout << "==================================================" << endl;*/
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
						if (interY < rightTopFrontCorner.y() && interY >= leftBottomBackCorner.y() && interZ < rightTopFrontCorner.z() && 
							interZ >= leftBottomBackCorner.z())
						{
							Intersections.emplace_back(V3d(leftBottomBackCorner.x(), interY, interZ));
							vector<int> nodes(4);
							nodes[0] = i;
							nodes[1] = i + m_resolution + 1;
							nodes[2] = i + 1;
							nodes[3] = nodes[1] + 1;
							relativeNodes.emplace_back(nodes);

							/*for (int lk = 0; lk < 4; lk++)
								cout << nodes[lk] << "        " << m_nodes.row(nodes[lk]) << endl;
							cout << V3d(leftBottomBackCorner.x(), interY, interZ) << endl;
							cout << "==================================================" << endl;*/
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
						if (interX < rightTopFrontCorner.x() && interX >= leftBottomBackCorner.x() && interZ < rightTopFrontCorner.z() &&
							interZ >= leftBottomBackCorner.z())
						{
							Intersections.emplace_back(V3d(interX, leftBottomBackCorner.y(), interZ));
							vector<int> nodes(4);
							nodes[0] = i;
							nodes[1] = i + square;
							nodes[2] = i + 1;
							nodes[3] = nodes[1] + 1;
							relativeNodes.emplace_back(nodes);

							/*for (int lk = 0; lk < 4; lk++)
								cout << nodes[lk] << "        " << m_nodes.row(nodes[lk]) << endl;
							cout << V3d(interX, leftBottomBackCorner.y(), interZ) << endl;
							cout << "==================================================" << endl;*/
						}
					}
				}
			}
		}
		std::cout << "\r Computing[" << std::fixed << std::setprecision(2) << ii * outpute << "%]";
		//int show_num = i / 50;
		/*for (int j = 1; j <= show_num; j++)
		{
			std::cout << "#";
			Sleep(10);
		}*/
	}
	std::cout << "#####################################################################." << std::endl;
}

double ThinShells::BaseFunction(const double& x, const double& node_x, const double& w)
{
	if (x <= node_x - w || x >= node_x + w) return 0.0;
	if (x <= node_x) return 1 + (x - node_x) / w;
	if (x > node_x) return 1 - (x - node_x) / w;
}

double ThinShells::BaseFunction4Point(const V3d& p, const V3d& nodePosition)
{
	double x = BaseFunction(p.x(), nodePosition.x(), m_width.x());
	if (x <= 0.0) return 0.0;
	double y = BaseFunction(p.y(), nodePosition.y(), m_width.y());
	if (y <= 0.0) return 0.0;
	double z = BaseFunction(p.z(), nodePosition.z(), m_width.z());
	if (z <= 0) return 0.0;
	return x * y * z;
}

VXd ThinShells::ImplicitField(const vector<V3d>& intersections, const vector<vector<int>>& relativeNodes)
{
	set<int> leafNodesID;
	std::cout << "Contructing the implicit field ..." << std::endl;
	int VN = m_V.rows();
	int interN = intersections.size();
	int N = VN + interN;
	double output = 100.0 / N;
	VXi I;
	MXd C, CC;
	VXd IBSValue(N);
	IBSValue.setZero();
	igl::signed_distance(m_nodes, m_V, m_F, igl::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER, m_BSplineCoefficient, I, C, CC);
	//cout << intersections.size() << "  " << relativeNodes.size() << endl;
	//cout << "width : " << m_width << endl;
	for (size_t i = 0; i < interN; i++)
	{
		std::cout << "\r Computing[" << std::fixed << std::setprecision(2) << i * output << "%]";
		for (int x : relativeNodes[i])
		{
			leafNodesID.insert(x);
			/*if (BaseFunction4Point(intersections[i], m_nodes.row(x)) == 0.0)
			{
				cout << "THERE!!!!!!!" << endl;
				cout << "inter: " << intersections[i].transpose() << "    node: " << m_nodes.row(x) << endl;
			}*/
			IBSValue[i] += BaseFunction4Point(intersections[i], m_nodes.row(x)) * m_BSplineCoefficient[x];
		}
	}

	int iout = 0;
	V3d beginV = m_nodes.colwise().minCoeff();
	
	int nxtRes = m_resolution + 1;
    int square = nxtRes * nxtRes;
	for (size_t i = 0; i < VN; i++)
	{
		std::cout << "\r Computing[" << std::fixed << std::setprecision(2) << (i + intersections.size()) * output << "%]";
		iout = i + interN;
		V3d p = modelVerts[i] - beginV;
		int xID = std::floor(p.x() / m_width.x());
		int yID = std::floor(p.y() / m_width.y());
		int zID = std::floor(p.z() / m_width.z());
		vector<int> nodesID;
		for (int a : {0, 1})
		{
			for (int b : {0, 1})
			{
				for (int c : {0, 1}) nodesID.emplace_back((xID + a) * square + (yID + b) * nxtRes + zID + c);
			}
		}
		//cout << m_verts[i] << endl;
		for (int x: nodesID)
		{
			//cout << x <<"     " << i << endl;
			//cout << std::fixed << std::setprecision(7) << m_nodes.row(x) << endl;
			leafNodesID.insert(x);
			 //if (BaseFunction4Point(m_verts[i], m_nodes.row(x)) == 0.0) cout << "THERE!!!!!!!" << endl;
			IBSValue[iout] += BaseFunction4Point(modelVerts[i], m_nodes.row(x)) * m_BSplineCoefficient[x];
		}
		//cout << "====================================" << endl;
	}

	cout << "#####################################################################." << endl;
	//m_shellValue = std::make_pair(IBSValue.minCoeff(), IBSValue.maxCoeff());
	m_shellValue.first = IBSValue.minCoeff();
	m_shellValue.second = IBSValue.maxCoeff();
	vector<V3d> MaxV;
	vector<V3i> MaxF;
	MarchingCubes mc1(m_BSplineCoefficient, m_nodes, m_shellValue.second, m_resolution, MaxV, MaxF);
	writeObjFile(OUT_PATH + m_modelName + "/" + std::to_string(m_resolution) + "MAX.obj", MaxV, MaxF);

	vector<V3d> ZeroV;
	vector<V3i> ZeroF;
	MarchingCubes mc2(m_BSplineCoefficient, m_nodes, 0, m_resolution, ZeroV, ZeroF);
	writeObjFile(OUT_PATH + m_modelName + "/" + std::to_string(m_resolution) + "ZERO.obj", ZeroV, ZeroF);

	vector<V3d> MinV;
	vector<V3i> MinF;
	MarchingCubes mc3(m_BSplineCoefficient, m_nodes, m_shellValue.first, m_resolution, MinV, MinF);
	writeObjFile(OUT_PATH + m_modelName + "/" + std::to_string(m_resolution) + "MIN.obj", MinV, MinF);

	return IBSValue;
}

MXd ThinShells::GetNodes() const
{
	return m_nodes;
}

V3d ThinShells::GetWidth() const
{
	return m_width;
}

VXd ThinShells::GetVIBSValueAtNodeVertex() const
{
	return m_BSplineCoefficient;
}

PDD ThinShells::GetThinShellValue() const
{
	return m_shellValue;
}

void ThinShells::SaveBValue2TXT(std::string filename, VXd X)
{
	std::ofstream out(filename);
	out << std::setiosflags(std::ios::fixed) << std::setprecision(9) << X << std::endl;
	out.close();
}
