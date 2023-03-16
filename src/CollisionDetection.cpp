#include "CollisionDetection.h"

vector<V3d> CollisionDetection::IntersectionOfCuboid(const vector<V3d>& cube1, const vector<V3d>& cube2)
{
	V3d min1 = cube1[0];
	V3d max1 = cube1[6];
	V3d max2 = cube2[6];
	V3d min2 = cube2[0];
	//if (min1.x() >= max2.x() || max1.x() <= min2.x() || min1.y() >= max2.y() || max1.y() <= min2.y() || min1.z() >= max2.z() || max1.z() <= min2.z()) return;
	double minX = min1.x() > min2.x() ? min1.x() : min2.x();
	double maxX = max1.x() < max2.x() ? max1.x() : max2.x();
	double minY = min1.y() > min2.y() ? min1.y() : min2.y();
	double maxY = max1.y() < max2.y() ? max1.y() : max2.y();
	double minZ = min1.z() > min2.z() ? min1.z() : min2.z();
	double maxZ = max1.z() < max2.z() ? max1.z() : max2.z();
	vector<V3d> newCube;
	newCube.emplace_back(V3d(minX, minY, minZ));
	newCube.emplace_back(V3d(maxX, minY, minZ));
	newCube.emplace_back(V3d(maxX, maxY, minZ));
	newCube.emplace_back(V3d(minX, maxY, minZ));
	newCube.emplace_back(V3d(minX, minY, maxZ));
	newCube.emplace_back(V3d(maxX, minY, maxZ));
	newCube.emplace_back(V3d(maxX, maxY, maxZ));
	newCube.emplace_back(V3d(minX, maxY, maxZ));
	return newCube;
}

void CollisionDetection::Initialize(const int& resolution1, const int& resolution2, const string& modelName1,
	const string& modelName2, const string& format1, const string& format2)
{
	m_is_collision = false;
	ThinShells model1(resolution1, modelName1, format1);
	ThinShells model2(resolution2, modelName2, format2);

	m_model1 = model1;
	m_model2 = model2;
}

double CollisionDetection::BaseFunction(const double& x, const double& node_x, const double& w)
{
	if (x <= node_x - w || x >= node_x + w) return 0.0;
	if (x <= node_x) return 1 + (x - node_x) / w;
	if (x > node_x) return 1 - (x - node_x) / w;
}

double CollisionDetection::BaseFunction4Point(const V3d& p, const V3d& nodePosition, const V3d& width)
{
	double x = BaseFunction(p.x(), nodePosition.x(), width.x());
	if (x <= 0.0) return 0.0;
	double y = BaseFunction(p.y(), nodePosition.y(), width.y());
	if (y <= 0.0) return 0.0;
	double z = BaseFunction(p.z(), nodePosition.z(), width.z());
	if (z <= 0) return 0.0;
	return x * y * z;
}

double CollisionDetection::ComputeSplineValue(const ThinShells& model, const V3d& p, const V3d& minBound)
{
	double value;
	int nxtRes1 = model.m_resolution + 1;
	int square1 = nxtRes1 * nxtRes1;
	int idx = std::floor((p.x() - minBound.x()) / model.m_width.x());
	int idy = std::floor((p.y() - minBound.y()) / model.m_width.y());
	int idz = std::floor((p.z() - minBound.z()) / model.m_width.z());
	double base = 0;
	if (idx == model.m_resolution)
	{
		if (idy == model.m_resolution)
		{
			if (idz == model.m_resolution)
			{
				value = model.m_BSplineCoefficient[nxtRes1 * square1 - 1];
				return value;
			}
			else
			{
				for (int c : {0, 1})
				{
					base += model.m_BSplineCoefficient[idx * square1 + idy * nxtRes1 + idz + c] *
						BaseFunction(p.z(), model.m_nodes.row(idz + c).z(), model.m_width.z());
				}
				return base;
			}
		}
		else
		{
			if (idz == model.m_resolution)
			{
				for (int b : {0, 1})
				{
					base += model.m_BSplineCoefficient[idx * square1 + (idy + b) * nxtRes1 + idz] *
						BaseFunction(p.y(), model.m_nodes.row((idy + b) * nxtRes1).y(), model.m_width.y());
				}
				return base;
			}
			else
			{
				for (int b : {0, 1})
				{
					for (int c : {0, 1})
					{
						int newID = idx * square1 + (idy + b) * nxtRes1 + idz + c;
						base += BaseFunction(p.y(), model.m_nodes.row(newID).y(), model.m_width.y()) *
							BaseFunction(p.z(), model.m_nodes.row(newID).z(), model.m_width.z()) * model.m_BSplineCoefficient[newID];
					}
				}
				return base;
			}
		}
	}

	else
	{
		if (idy == model.m_resolution)
		{
			if (idz == model.m_resolution)
			{
				for (int a : {0, 1})
				{
					int newID = (idx + a) * square1 + idy * nxtRes1 + idz;
					base += model.m_BSplineCoefficient[newID] * BaseFunction(p.x(), model.m_nodes.row(newID).x(), model.m_width.x());
				}
				return base;
			}
			else
			{
				for (int a : {0, 1})
				{
					for (int c : {0, 1})
					{
						int newID = (idx + a) * square1 + idy * nxtRes1 + idz + c;
						base += BaseFunction(p.x(), model.m_nodes.row(newID).x(), model.m_width.x()) *
							BaseFunction(p.z(), model.m_nodes.row(newID).z(), model.m_width.z()) * model.m_BSplineCoefficient[newID];
					}
				}
				return base;
			}
		}
		else
		{
			if (idz == m_model1.m_resolution)
			{
				for (int a : {0, 1})
				{
					for (int b : {0, 1})
					{
						int newID = (idx + a) * square1 + (idy + b) * nxtRes1 + idz;
						base += BaseFunction(p.x(), model.m_nodes.row(newID).x(), model.m_width.x()) *
							BaseFunction(p.y(), model.m_nodes.row(newID).y(), model.m_width.y()) * model.m_BSplineCoefficient[newID];
					}
				}
				return base;
			}
			for (int a : {0, 1})
			{
				for (int b : {0, 1})
				{
					for (int c : {0, 1})
					{
						int newID = (idx + a) * square1 + (idy + b) * nxtRes1 + idz + c;
						/*if (newID >= m_model1.m_BSplineCoefficient.size())
						{
							cout << min1.x() << "  " << p.x() << "  " << max1.x() << "  ";
							cout << idx + a << " " << idy + b << "  " << idz + c << " ";
							cout << newID << "   " << m_model1.m_BSplineCoefficient.size() << endl;
						}*/
						base += BaseFunction4Point(p, model.m_nodes.row(newID), model.m_width) * model.m_BSplineCoefficient[newID];
					}
				}
			}
			return base;
		}
	}
}

void CollisionDetection::ExtractCommonNodes(vector<V3d>& commonNodesVertex, vector<vector<int>>& commonNodesID, vector<double>& value1, vector<double>& value2)
{
	int nxtRes1 = m_model1.m_resolution + 1;
	int nxtRes2 = m_model2.m_resolution + 1;
	int square1 = nxtRes1 * nxtRes1;
	int square2 = nxtRes2 * nxtRes2;
	V3d min1 = m_model1.m_nodes.colwise().minCoeff();
	V3d max1 = m_model1.m_nodes.colwise().maxCoeff();
	V3d min2 = m_model2.m_nodes.colwise().minCoeff();
	V3d max2 = m_model2.m_nodes.colwise().maxCoeff();
	cout << " node1: " << min1.transpose() << "   " << max1.transpose() << endl;
	cout << " ndoe2: " << min2.transpose() << "   " << max2.transpose() << endl;
	
	// the two models can not collide
	if (min1.x() >= max2.x() || min2.x() >= max1.x()) return;
	if (min1.y() >= max2.y() || min2.y() >= max1.y()) return;
	if (min1.z() >= max2.z() || min2.z() >= max1.z()) return;

	// the intersect region of two type nodes
	std::pair<V3d, V3d> bound;
	bound.first.x() = min1.x() > min2.x() ? min1.x() : min2.x();
	bound.first.y() = min1.y() > min2.y() ? min1.y() : min2.y();
	bound.first.z() = min1.z() > min2.z() ? min1.z() : min2.z();
	bound.second.x() = max1.x() < max2.x() ? max1.x() : max2.x();
	bound.second.y() = max1.y() < max2.y() ? max1.y() : max2.y();
	bound.second.z() = max1.z() < max2.z() ? max1.z() : max2.z();
	cout << bound.first.transpose() << "   " << bound.second.transpose() << endl;
	
	// ********************************* split nodes inside the bound into several cubiods with various sizes   *********
	set<double> X;
	set<double> Y;
	set<double> Z;
	X.insert(bound.first.x());
	X.insert(bound.second.x());
	int idx11 = std::floor((bound.first.x() - min1.x()) / m_model1.m_width.x()) + 1;
	int idx12 = std::floor((bound.second.x() - min1.x()) / m_model1.m_width.x());
	int idx21 = std::floor((bound.first.x() - min2.x()) / m_model2.m_width.x()) + 1;
	int idx22 = std::floor((bound.second.x() - min2.x()) / m_model2.m_width.x());
	for (int i = idx11; i <= idx12; i++) X.insert(m_model1.m_nodes.row(i * square1).x());
	for (int i = idx21; i <= idx22; i++) X.insert(m_model2.m_nodes.row(i * square2).x());
	
	/*cout << "bound X : " << std::fixed << std::setprecision(5) << bound.first.x() << "   " << bound.second.x() << endl;
	for (double x : X) cout << x << "  ";
	cout << endl;*/

	Y.insert(bound.first.y());
	Y.insert(bound.second.y());
	int idy11 = std::floor((bound.first.y() - min1.y()) / m_model1.m_width.y()) + 1;
	int idy12 = std::floor((bound.second.y() - min1.y()) / m_model1.m_width.y());
	int idy21 = std::floor((bound.first.y() - min2.y()) / m_model2.m_width.y()) + 1;
	int idy22 = std::floor((bound.second.y() - min2.y()) / m_model2.m_width.y());
	for (int i = idy11; i <= idy12; i++) Y.insert(m_model1.m_nodes.row(i * nxtRes1).y());
	for (int i = idy21; i <= idy22; i++) Y.insert(m_model2.m_nodes.row(i * nxtRes2).y());
	
	/*cout << "bound Y : " << std::fixed << std::setprecision(5) << bound.first.y() << "   " << bound.second.y() << endl;
	for (double y : Y) cout << y << "  ";
	cout << endl;*/
	
	Z.insert(bound.first.z());
	Z.insert(bound.second.z());
	int idz11 = std::floor((bound.first.z() - min1.z()) / m_model1.m_width.z()) + 1;
	int idz12 = std::floor((bound.second.z() - min1.z()) / m_model1.m_width.z());
	int idz21 = std::floor((bound.first.z() - min2.z()) / m_model2.m_width.z()) + 1;
	int idz22 = std::floor((bound.second.z() - min2.z()) / m_model2.m_width.z());
	for (int i = idz11; i <= idz12; i++) Z.insert(m_model1.m_nodes.row(i).z());
	for (int i = idz21; i <= idz22; i++) Z.insert(m_model2.m_nodes.row(i).z());

	/*cout << "bound Z : " << std::fixed << std::setprecision(5) << bound.first.z() << "   " << bound.second.z() << endl;
	for (double z : Z) cout << z << "  ";
	cout << endl;*/

	int NX = X.size();
	int NY = Y.size();
	int NZ = Z.size();
	int NYZ = NY * NZ;
	vector<V3d> overlappingNodesVertices;
	for (double x : X)
	{
		for (double y : Y)
		{
			for (double z : Z) overlappingNodesVertices.emplace_back(V3d(x, y, z));
		}
	}
	
	/*cout << "size = " << overlappingNodesVertices.size() << endl;
	cout << NX * NY * NZ << endl;*/
	vector<vector<int>> overlappingNodesID;
	for (int i = 0; i < NX - 1; i++)
	{
		for (int j = 0; j < NY - 1; j++)
		{
			for (int k = 0; k < NZ - 1; k++)
			{
				vector<int> v;
				v.emplace_back(i * NYZ + j * NZ + k);
				v.emplace_back((i + 1) * NYZ + j * NZ + k);
				v.emplace_back((i + 1) * NYZ + (j + 1) * NZ + k);
				v.emplace_back(i * NYZ  + (j + 1) * NZ + k);
				v.emplace_back(i * NYZ + j * NZ + k + 1);
				v.emplace_back((i + 1) * NYZ + j * NZ + k + 1);
				v.emplace_back((i + 1) * NYZ + (j + 1) * NZ + k + 1);
				v.emplace_back(i * NYZ + (j + 1) * NZ + k + 1);
				overlappingNodesID.emplace_back(v);
			}
		}
	}
	cout << overlappingNodesID.size() << endl;
	SaveNode(OUT_DIR + m_model1.m_modelName + "/" + std::to_string(m_model1.m_resolution) + "SplitedNodes.obj", overlappingNodesID, overlappingNodesVertices);

	// *************** Extract the nodes that the first model passes through*************************
	VXd IBSvalueInNodes1(overlappingNodesVertices.size());   // the IBS value of splited node vertices in the first ndoes
	for (int i = 0; i < overlappingNodesVertices.size(); i++)
	{
		V3d p = overlappingNodesVertices[i];
		IBSvalueInNodes1[i] = ComputeSplineValue(m_model1, p, min1);
	}

	//cout << IBSvalueInNodes1 << endl;
	vector<vector<int>> passNodesID;
	double minV;
	double maxV;
	set<int> passVertexID;     // the id of some overlapping nodes vertices that contain the first model's 0-level surface
	for (int i = 0; i < overlappingNodesID.size(); i++)
	{
		vector<int> v = overlappingNodesID[i];
		minV = IBSvalueInNodes1[v[0]];
		maxV = IBSvalueInNodes1[v[0]];
		for (int j = 1; j < 8; j++)
		{
			minV = minV < IBSvalueInNodes1[v[j]] ? minV : IBSvalueInNodes1[v[j]];
			maxV = maxV > IBSvalueInNodes1[v[j]] ? maxV : IBSvalueInNodes1[v[j]];
		}
		if (minV * maxV <= 0)
		{
			passNodesID.emplace_back(v);
			for (int j = 0; j < 8; j++) passVertexID.insert(v[j]);
		}
	}

	// the fact that the first model does not pass the intersect node region means two models will not collide
	if (passVertexID.empty()) return;

	SaveNode(OUT_DIR + m_model1.m_modelName + "/" + std::to_string(m_model1.m_resolution) + "ContainedNodes.obj", passNodesID, overlappingNodesVertices);

	// *************** Extract the nodes that both model pass through ****************************
	vector<double> IBSvalueInNodes2(overlappingNodesVertices.size(), 0);   // the IBS value of splited node vertices in the second ndoes
	int idx, idy, idz;
	double base;
	for (int i : passVertexID)
	{
		V3d p = overlappingNodesVertices[i];
		IBSvalueInNodes2[i] = ComputeSplineValue(m_model2, p, min2);
	}

	set<int> commonVertexID;
	vector<vector<int>> commonNodesID2;
	for (int i = 0; i < passNodesID.size(); i++)
	{
		vector<int> v = passNodesID[i];
		minV = IBSvalueInNodes2[v[0]];
		maxV = IBSvalueInNodes2[v[0]];
		for (int j = 1; j < 8; j++)
		{
			minV = minV < IBSvalueInNodes2[v[j]] ? minV : IBSvalueInNodes2[v[j]];
			maxV = maxV > IBSvalueInNodes2[v[j]] ? maxV : IBSvalueInNodes2[v[j]];
		}
		if (minV * maxV > 0)
		{
			m_is_collision = true;
			continue;
		}
		commonNodesID2.emplace_back(v);
		for (int j = 0; j < 8; j++) commonVertexID.insert(v[j]);
	}
	if (!m_is_collision) return;

	m_is_collision = false;
	// delete the superfluous nodes and relative vertices
	std::map<int, int> swapid;
	int count = 0;
	for (int i : commonVertexID)
	{
		value1.emplace_back(IBSvalueInNodes1[i]);
		value2.emplace_back(IBSvalueInNodes2[i]);
		commonNodesVertex.emplace_back(overlappingNodesVertices[i]);
		swapid[i] = count;
		count++;
	}

	for (vector<int> v : commonNodesID2)
	{
		vector<int> newv;
		for (int i : v)
		{
			newv.emplace_back(swapid[i]);
		}
		commonNodesID.emplace_back(newv);
	}
	
	SaveNode(OUT_DIR + m_model2.m_modelName + "/" + std::to_string(m_model2.m_resolution) + "CommonNodes.obj", commonNodesID, commonNodesVertex);
	SaveNode(OUT_DIR + m_model1.m_modelName + "/" + std::to_string(m_model1.m_resolution) + "CommonNodes.obj", commonNodesID, commonNodesVertex);
}

vector<std::pair<V3d, V3d>> CollisionDetection::ExtractInterLinesInSingleNode(const vector<V3d>& verts, const vector<int>& node, 
	const vector<double>& val1, const vector<double>& val2)
{
	// Q : 为什么0等值面必然与node边有交？  bingo√
	vector<std::pair<V3d, V3d>> intersectingLines;

	vector<bool> is_interWithEdge1(12, false);        // judge whether 0-level surface in the first is intersecting the edge[i] for i in range(12)
	vector<bool> is_interWithEdge2(12, false);        // judge whether 0-level surface in the second is intersecting the edge[i] for i in range(12)
	vector<V3d> intersections(12, V3d(100, 100, 100));
	vector<double> val14Inters(12, 100);             // the IBS value of the intersection in the first function field

	// compute the possible intersections for every edge in the second field
	for (int i = 0; i < 12; i++)                     // there exists  only one intersection between the surface and the edge at most
	{
		int p1 = node[m_nodeEdges[i][0]];
		int p2 = node[m_nodeEdges[i][1]];
		if (val2[p1] * val2[p2] > 0) continue;
		double mu = val2[p2] / (val2[p2] - val2[p1]);
		val14Inters[i] = (1 - mu) * val1[p2] + mu * val1[p1];
		intersections[i] = (1 - mu) * verts[p2] + mu * verts[p1];
		is_interWithEdge2[i] = true;
	}

	// extract faces whose may intersect surface
	set<int> facePossibleHaveInter;
	for (int i = 0; i < 12; i++)
	{
		int p1 = node[m_nodeEdges[i][0]];
		int p2 = node[m_nodeEdges[i][1]];
		if (val1[p1] * val1[p2] > 0) continue;
		is_interWithEdge1[i] = true;
		facePossibleHaveInter.insert(m_edgeFace[i][0]);
		facePossibleHaveInter.insert(m_edgeFace[i][1]);
	}

	for (int i : facePossibleHaveInter)
	{
		vector<PII> edgeIDInFace1;  // first : edge ID in node, second : edge ID in face
		vector<PII> edgeIDInFace2;
		int count = 0;
		for (int j : m_faceEdge[i])
		{
			if (is_interWithEdge1[j]) edgeIDInFace1.emplace_back(make_pair(j, count));
			if (is_interWithEdge2[j]) edgeIDInFace2.emplace_back(make_pair(j, count));
			count++;
		}

		if (edgeIDInFace2.size() == 0) continue;
		if (edgeIDInFace2.size() == 2)
		{
			if (val14Inters[edgeIDInFace2[0].first] == 0 && val14Inters[edgeIDInFace2[1].first] == 0)
			{
				if (edgeIDInFace1.size() == 2)
				{
					intersectingLines.emplace_back(std::make_pair(intersections[edgeIDInFace2[0].first], intersections[edgeIDInFace2[1].first]));
					continue;
				}
				if (edgeIDInFace1.size() == 4)
				{
					if (std::abs(edgeIDInFace2[0].second - edgeIDInFace2[1].second) != 1) continue;

				}
				
			}
			/*if (val14Inters[edgeIDInFace2[0]] * val14Inters[edgeIDInFace2[1]] > 0)
			{
				if (edgeIDInFace1.size() == 2) continue;
				if (edgeIDInFace1.size() == 4)
				{
					int p1 = edgeIDInFace1[0];
					int p2 = edgeIDInFace1[1];
					double mu = val1[p2] / (val1[p2] - val1[p1]);
					val14Inters[i] = (1 - mu) * val1[p2] + mu * val1[p1];
					intersections[i] = (1 - mu) * verts[p2] + mu * verts[p1];
				}
			}*/
			

		}
		if (edgeIDInFace2.size() == 4)
		{
			cout << "IT'S POSSIBLE!!!!!!!!!" << endl;
		}
	}
	return intersectingLines;
}

void CollisionDetection::ExtractIntersectLines(const int& resolution1, const int& resolution2, const string& modelName1,
	const string& modelName2, const string& format1, const string& format2)
{
	vector<PII> intersectLines;
	vector<V3d> commonNodesVertex;
	vector<vector<int>> commonNodesID;
	vector<double> value1;
	vector<double> value2;
	Initialize(resolution1, resolution2, modelName1, modelName2, format1, format2);
	cout << m_model1.m_nodes.rows() << "   " << m_model1.m_BSplineCoefficient.size() << endl;
	ExtractCommonNodes(commonNodesVertex, commonNodesID, value1, value2);
	for (int i = 0; i < commonNodesID.size(); i++)
	{
		vector<int> v = commonNodesID[i];
		ExtractInterLinesInSingleNode(commonNodesVertex, v, value1, value2);
	}
}

void CollisionDetection::SaveNode(const string& filename, const vector<vector<int>>& overlappingNodesID, const MXd& nodes) const
{
	std::ofstream out(filename);
	int count = -8;
	for (int i = 0; i < overlappingNodesID.size(); i++)
	{
		vector<V3d> corners;
		corners.resize(8);
		for (int j = 0; j < 8; j++)
		{
			corners[j] = nodes.row(overlappingNodesID[i][j]);
			out << "v " << std::setiosflags(std::ios::fixed) << std::setprecision(9) << corners[j].transpose() << endl;
			count++;
		}

		out << "l " << 1 + count << " " << 2 + count << endl;
		out << "l " << 1 + count << " " << 4 + count << endl;
		out << "l " << 1 + count << " " << 5 + count << endl;
		out << "l " << 2 + count << " " << 3 + count << endl;
		out << "l " << 2 + count << " " << 6 + count << endl;
		out << "l " << 3 + count << " " << 4 + count << endl;
		out << "l " << 3 + count << " " << 7 + count << endl;
		out << "l " << 4 + count << " " << 8 + count << endl;
		out << "l " << 5 + count << " " << 6 + count << endl;
		out << "l " << 5 + count << " " << 8 + count << endl;
		out << "l " << 6 + count << " " << 7 + count << endl;
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

void CollisionDetection::SaveNode(const string& filename, const vector<vector<int>>& overlappingNodesID, const vector<V3d>& nodes) const
{
	std::ofstream out(filename);
	int count = -8;
	for (int i = 0; i < overlappingNodesID.size(); i++)
	{
		vector<V3d> corners;
		corners.resize(8);
		for (int j = 0; j < 8; j++)
		{
			corners[j] = nodes[overlappingNodesID[i][j]];
			out << "v " << std::setiosflags(std::ios::fixed) << std::setprecision(9) << corners[j].transpose() << endl;
			count++;
		}

		out << "l " << 1 + count << " " << 2 + count << endl;
		out << "l " << 1 + count << " " << 4 + count << endl;
		out << "l " << 1 + count << " " << 5 + count << endl;
		out << "l " << 2 + count << " " << 3 + count << endl;
		out << "l " << 2 + count << " " << 6 + count << endl;
		out << "l " << 3 + count << " " << 4 + count << endl;
		out << "l " << 3 + count << " " << 7 + count << endl;
		out << "l " << 4 + count << " " << 8 + count << endl;
		out << "l " << 5 + count << " " << 6 + count << endl;
		out << "l " << 5 + count << " " << 8 + count << endl;
		out << "l " << 6 + count << " " << 7 + count << endl;
		out << "l " << 7 + count << " " << 8 + count << endl;
	}
	out.close();
}
