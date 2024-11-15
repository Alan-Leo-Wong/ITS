#include "BaseModel.h"
#include "utils/IO.hpp"
#include "utils/Common.hpp"
#include "utils/String.hpp"
#include "cuAcc/CUDACompute.h"
#include <sstream>
#include <iomanip>
#include <igl/writeOBJ.h>
#include <igl/read_triangle_mesh.h>
#include <igl/per_vertex_normals.h>
#include <igl/point_mesh_squared_distance.h>

//////////////////////
//   Model  Utils   //
//////////////////////
void BaseModel::setBoundingBox(const float& scaleSize)
{
	V3d minV = m_V.colwise().minCoeff();
	V3d maxV = m_V.colwise().maxCoeff();

	modelBoundingBox = AABox(minV, maxV);
}

void BaseModel::setUniformBoundingBox()
{
	V3d minV = m_V.colwise().minCoeff();
	V3d maxV = m_V.colwise().maxCoeff();

	modelBoundingBox = AABox<V3d>(minV, maxV); // initialize answer
	Eigen::Vector3d lengths = maxV - minV; // check length of given bbox in every direction
	float max_length = fmaxf(lengths.x(), fmaxf(lengths.y(), lengths.z())); // find max length
	for (unsigned int i = 0; i < 3; i++) { // for every direction (X,Y,Z)
		if (max_length == lengths[i]) {
			continue;
		}
		else {
			float delta = max_length - lengths[i]; // compute difference between largest length and current (X,Y or Z) length
			modelBoundingBox.boxOrigin[i] = minV[i] - (delta / 2.0f); // pad with half the difference before current min
			modelBoundingBox.boxEnd[i] = maxV[i] + (delta / 2.0f); // pad with half the difference behind current max
		}
	}

	// Next snippet adresses the problem reported here: https://github.com/Forceflow/cuda_voxelizer/issues/7
	// Suspected cause: If a triangle is axis-aligned and lies perfectly on a voxel edge, it sometimes gets counted / not counted
	// Probably due to a numerical instability (division by zero?)
	// Ugly fix: we pad the bounding box on all sides by 1/10001th of its total length, bringing all triangles ever so slightly off-grid
	Eigen::Vector3d epsilon = (modelBoundingBox.boxEnd - modelBoundingBox.boxOrigin) / 11; // 之前是10001
	modelBoundingBox.boxOrigin -= epsilon;
	modelBoundingBox.boxEnd += epsilon;
	modelBoundingBox.boxWidth = modelBoundingBox.boxEnd - modelBoundingBox.boxOrigin;
}

// 用于to unit cube
Eigen::Matrix4d BaseModel::calcUnitCubeTransformMatrix()
{
	Eigen::RowVector3d boxMin = m_V.colwise().minCoeff();
	Eigen::RowVector3d boxMax = m_V.colwise().maxCoeff();

	// Get the target solveRes (along the largest dimension)
	double scale = boxMax[0] - boxMin[0];
	double minScale = scale;
	for (int d = 1; d < 3; d++)
	{
		scale = std::max<double>(scale, boxMax[d] - boxMin[d]);
		minScale = std::min<double>(scale, boxMax[d] - boxMin[d]);
	}
	// std::cout << 1.1 + scale / minScale << std::endl;
	// scaleFactor =
	scale *= scaleFactor;
	Eigen::Vector3d center = 0.5 * boxMax + 0.5 * boxMin;

	for (int i = 0; i < 3; i++)
		center[i] -= scale / 2;
	Eigen::Matrix4d zoomMatrix = Eigen::Matrix4d::Identity();
	Eigen::Matrix4d transMatrix = Eigen::Matrix4d::Identity();
	for (int i = 0; i < 3; i++)
	{
		zoomMatrix(i, i) = 1. / scale;
		transMatrix(3, i) = -center[i];
	}
	return zoomMatrix * transMatrix;
}

void BaseModel::model2UnitCube()
{
	auto transMat = calcUnitCubeTransformMatrix();
	for (int i = 0; i < m_V.rows(); ++i)
	{
		m_V.row(i) += transMat.block(3, 0, 1, 3);
		m_V.row(i) = m_V.row(i) * transMat.block(0, 0, 3, 3);
	}
}

void BaseModel::unitCube2Model()
{
	auto transMat = calcUnitCubeTransformMatrix();
	Eigen::Matrix3d inverseTrans = transMat.block(0, 0, 3, 3).inverse();
	for (int i = 0; i < m_V.rows(); ++i)
	{
		m_V.row(i) = m_V.row(i) * inverseTrans;
		m_V.row(i) -= transMat.block(3, 0, 1, 3);
	}
}

// 用于缩放模型
Eigen::Matrix3d BaseModel::calcScaleMatrix()
{
	Eigen::RowVector3d boxMin = m_V.colwise().minCoeff();
	Eigen::RowVector3d boxMax = m_V.colwise().maxCoeff();

	// Get the target solveRes (along the largest dimension)
	double scale = boxMax[0] - boxMin[0];
	scale *= scaleFactor;
	Eigen::Matrix3d zoomMatrix;
	for (int i = 0; i < 3; i++)
		zoomMatrix(i, i) = scale;
	return zoomMatrix;
}

void BaseModel::addNoise(const double& noisePercentage, const double& min_val, const double& max_val)
{
	// 计算需要扰动的节点数量
	int numNodes = m_V.rows();
	int numNoisyNodes = static_cast<int>(noisePercentage * numNodes);

	// 设置随机数生成器
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(min_val, max_val); // 设置扰动范围为 [-0.1, 0.1]

	// 随机扰动节点位置
	for (int i = 0; i < numNoisyNodes; i++) 
	{
		// 随机选择一个节点索引
		int nodeIndex = std::rand() % numNodes;

		// 在节点位置上添加随机扰动
		m_V.row(nodeIndex) += dis(gen) * Eigen::RowVector3d::Random();
	}
}

void BaseModel::zoomModel()
{
	Eigen::Matrix3d zoomMat = calcScaleMatrix();
	m_V = m_V * zoomMat;
}

// 用于一般的transform，先缩放模型，再将新的中心点移至缩放前模型的中心点
Eigen::Matrix4d BaseModel::calcTransformMatrix(const float& _scaleFactor)
{
	Eigen::RowVector3d boxMin = m_V.colwise().minCoeff();
	Eigen::RowVector3d boxMax = m_V.colwise().maxCoeff();

	// Get the target solveRes (along the largest dimension)
	double scale = boxMax[0] - boxMin[0];
	for (int d = 1; d < 3; d++)
		scale = std::max<double>(scale, boxMax[d] - boxMin[d]);
	if (_scaleFactor > 1e-9) scale *= _scaleFactor;
	else scale *= scaleFactor;
	Eigen::Vector3d center = 0.5 * boxMax + 0.5 * boxMin;

	for (int i = 0; i < 3; i++)
		center[i] -= scale / 2;
	Eigen::Matrix4d zoomMatrix = Eigen::Matrix4d::Identity();
	Eigen::Matrix4d transMatrix = Eigen::Matrix4d::Identity();
	for (int i = 0; i < 3; i++)
	{
		zoomMatrix(i, i) = scaleFactor;
		transMatrix(3, i) = center[i] * (scaleFactor - 1);
	}
	return zoomMatrix * transMatrix;
}

void BaseModel::transformModel(const float& _scaleFactor)
{
	Eigen::Matrix4d transMat = calcTransformMatrix(_scaleFactor);
	for (int i = 0; i < m_V.rows(); ++i)
	{
		m_V.row(i) += transMat.block(3, 0, 1, 3);
		m_V.row(i) = m_V.row(i) * transMat.block(0, 0, 3, 3);
	}
}

Eigen::MatrixXd BaseModel::generateGaussianRandomPoints(const size_t& numPoints, const float& _scaleFactor, const float& dis)
{
	Eigen::MatrixXd M;
	const Eigen::RowVector3d min_area = modelBoundingBox.boxOrigin;
	const Eigen::RowVector3d max_area = modelBoundingBox.boxEnd;
	getGaussianRandomMatrix<double>(min_area, max_area, numPoints, _scaleFactor, dis, M);
	return M;
}

Eigen::MatrixXd BaseModel::generateUniformRandomPoints(const size_t& numPoints, const float& _scaleFactor, const float& dis)
{
	Eigen::MatrixXd M;
	const Eigen::RowVector3d min_area = modelBoundingBox.boxOrigin;
	const Eigen::RowVector3d max_area = modelBoundingBox.boxEnd;
	getUniformRandomMatrix<double>(min_area, max_area, numPoints, _scaleFactor, dis, M);
	return M;
}

Eigen::MatrixXd BaseModel::generateGaussianRandomPoints(const string& filename, const size_t& numPoints,
	const float& _scaleFactor, const float& dis)
{
	Eigen::MatrixXd M;
	const Eigen::RowVector3d min_area = modelBoundingBox.boxOrigin;
	const Eigen::RowVector3d max_area = modelBoundingBox.boxEnd;
	getGaussianRandomMatrix<double>(min_area, max_area, numPoints, _scaleFactor, dis, M);

	checkDir(filename);
	std::ofstream out(filename, std::ofstream::out);
	if (!out) { fprintf(stderr, "[I/O] Error: File %s could not be opened!", filename.c_str()); return M; }
	cout << "-- Save random points to " << std::quoted(filename) << endl;

	//std::cout << getFileExtension(filename) << std::endl;
	if (getFileExtension(filename) == ".obj")
		gvis::writePointCloud(M, out);
	else if (getFileExtension(filename) == ".xyz")
		gvis::writePointCloud_xyz(M, out);

	return M;
}

Eigen::MatrixXd BaseModel::generateUniformRandomPoints(const string& filename, const size_t& numPoints,
	const float& _scaleFactor, const float& dis)
{
	Eigen::MatrixXd M;
	const Eigen::RowVector3d min_area = modelBoundingBox.boxOrigin;
	const Eigen::RowVector3d max_area = modelBoundingBox.boxEnd;
	getUniformRandomMatrix<double>(min_area, max_area, numPoints, _scaleFactor, dis, M);

	checkDir(filename);
	std::ofstream out(filename, std::ofstream::out);
	if (!out) { fprintf(stderr, "[I/O] Error: File %s could not be opened!", filename.c_str()); return M; }
	cout << "-- Save random points to " << std::quoted(filename) << endl;

	//std::cout << getFileExtension(filename) << std::endl;
	if (getFileExtension(filename) == ".obj")
		gvis::writePointCloud(M, out);
	else if (getFileExtension(filename) == ".xyz")
		gvis::writePointCloud_xyz(M, out);

	return M;
}

vector<V3d> BaseModel::generateUniformRandomPoints(const string& filename, const size_t& numPoints,
	const double& _scaleFactor, const V3d& dis)
{
	vector<V3d> randomPoints;
	getUniformRandomMatrix<V3d>(modelBoundingBox, numPoints, _scaleFactor, dis, randomPoints);

	checkDir(filename);
	std::ofstream out(filename, std::ofstream::out);
	if (!out) { fprintf(stderr, "[I/O] Error: File %s could not be opened!", filename.c_str()); return randomPoints; }
	cout << "-- Save random points to " << std::quoted(filename) << endl;

	//std::cout << getFileExtension(filename) << std::endl;
	if (getFileExtension(filename) == ".obj")
		gvis::writePointCloud(randomPoints, out);
	else if (getFileExtension(filename) == ".xyz")
		gvis::writePointCloud_xyz(randomPoints, out);

	return randomPoints;
}

vector<V3d> BaseModel::generateGaussianRandomPoints(const string& filename, const size_t& numPoints,
	const double& _scaleFactor, const V3d& dis)
{
	vector<V3d> randomPoints;
	getGaussianRandomMatrix<V3d>(modelBoundingBox, numPoints, _scaleFactor, dis, randomPoints);

	checkDir(filename);
	std::ofstream out(filename, std::ofstream::out);
	if (!out) { fprintf(stderr, "[I/O] Error: File %s could not be opened!", filename.c_str()); return randomPoints; }
	cout << "-- Save random points to " << std::quoted(filename) << endl;

	//std::cout << getFileExtension(filename) << std::endl;
	if (getFileExtension(filename) == ".obj")
		gvis::writePointCloud(randomPoints, out);
	else if (getFileExtension(filename) == ".xyz")
		gvis::writePointCloud_xyz(randomPoints, out);

	return randomPoints;
}

void BaseModel::setTriAttributes()
{
	cuAcc::launch_modelTriAttributeKernel(nModelTris, modelTris);
}

vector<V2i> BaseModel::extractEdges()
{
	cout << "Extracting edges from " << std::quoted(modelName) << endl;

	vector<V2i> edges;
	set<PII> uset;

	for (V3i f : modelFaces)
	{
		int maxF = f.maxCoeff();
		int minF = f.minCoeff();
		int middleF = f.sum() - maxF - minF;

		uset.insert(std::make_pair(minF, middleF));
		uset.insert(std::make_pair(middleF, maxF));
		uset.insert(std::make_pair(minF, maxF));
	}
	for (PII it : uset)
		edges.emplace_back(V2i(it.first, it.second));

	cout << "-- Number of " << modelName << "'edges: " << edges.size() << endl;
	return edges;
}

vector<vector<V3d>> BaseModel::extractIsoline(const vector<double>& scalarField, const double& val)const
{
	map<PII, V3d > pointsOnEdges;  // 根据边的pair值索引交点的位置
	map<PII, set<PII>> fromOneEdgePoint2Neighbors;   // key值为边上点，value表示相邻的两个点
	for (V3i t : modelFaces)
	{
		int id1 = t.x();
		int id2 = t.y();
		int id3 = t.z();
		V3d p1 = modelVerts[id1];
		V3d p2 = modelVerts[id2];
		V3d p3 = modelVerts[id3];
		double scalar1 = scalarField[id1];
		double scalar2 = scalarField[id2];
		double scalar3 = scalarField[id3];

		// 等值线与某条边重合
		if (scalar1 == val && scalar2 == val)
		{
			if (pointsOnEdges.find(make_pair(id1, id1)) == pointsOnEdges.end())
			{
				pointsOnEdges[make_pair(id1, id1)] = p1;
			}
			fromOneEdgePoint2Neighbors[make_pair(id1, id1)].insert(make_pair(id2, id2));
			if (pointsOnEdges.find(make_pair(id2, id2)) == pointsOnEdges.end())
			{
				pointsOnEdges[make_pair(id2, id2)] = p2;
			}
			fromOneEdgePoint2Neighbors[make_pair(id2, id2)].insert(make_pair(id1, id1));
		}
		else if (scalar2 == val && scalar3 == val)
		{
			if (pointsOnEdges.find(make_pair(id2, id2)) == pointsOnEdges.end())
			{
				pointsOnEdges[make_pair(id2, id2)] = p2;
			}
			fromOneEdgePoint2Neighbors[make_pair(id2, id2)].insert(make_pair(id3, id3));
			if (pointsOnEdges.find(make_pair(id3, id3)) == pointsOnEdges.end())
			{
				pointsOnEdges[make_pair(id3, id3)] = p3;
			}
			fromOneEdgePoint2Neighbors[make_pair(id3, id3)].insert(make_pair(id2, id2));
		}
		else if (scalar3 == val && scalar1 == val)
		{
			if (pointsOnEdges.find(make_pair(id3, id3)) == pointsOnEdges.end())
			{
				pointsOnEdges[make_pair(id3, id3)] = p3;
			}
			fromOneEdgePoint2Neighbors[make_pair(id3, id3)].insert(make_pair(id1, id1));
			if (pointsOnEdges.find(make_pair(id1, id1)) == pointsOnEdges.end())
			{
				pointsOnEdges[make_pair(id1, id1)] = p1;
			}
			fromOneEdgePoint2Neighbors[make_pair(id1, id1)].insert(make_pair(id3, id3));
		}
		else
		{
			vector<PII>resPoint;   // 存储穿过某条边的交点所在边的两端端点id
			bool flags[3] = { false };       // 用于判断在哪条边上有交
			// 等值线穿过边
			if ((scalar1 > val) != (scalar2 > val))
			{
				if (fromOneEdgePoint2Neighbors.find(make_pair(min(id1, id2), max(id1, id2))) == fromOneEdgePoint2Neighbors.end())
				{
					double proportion = (scalar1 - val) / (scalar1 - scalar2);
					V3d intersection = proportion * p2 + (1 - proportion) * p1;
					pointsOnEdges[make_pair(min(id1, id2), max(id1, id2))] = intersection;
				}
				resPoint.emplace_back(make_pair(min(id1, id2), max(id1, id2)));
				flags[2] = true;
			}
			if ((scalar2 > val) != (scalar3 > val))
			{
				if (fromOneEdgePoint2Neighbors.find(make_pair(min(id2, id3), max(id2, id3))) == fromOneEdgePoint2Neighbors.end())
				{
					double proportion = (scalar2 - val) / (scalar2 - scalar3);
					V3d intersection = proportion * p3 + (1 - proportion) * p2;
					pointsOnEdges[make_pair(min(id2, id3), max(id2, id3))] = intersection;
				}
				resPoint.emplace_back(make_pair(min(id2, id3), max(id2, id3)));
				flags[0] = true;
			}
			if ((scalar3 > val) != (scalar1 > val))
			{
				if (fromOneEdgePoint2Neighbors.find(make_pair(min(id3, id1), max(id3, id1))) == fromOneEdgePoint2Neighbors.end())
				{
					double proportion = (scalar3 - val) / (scalar3 - scalar1);
					V3d intersection = proportion * p1 + (1 - proportion) * p3;
					pointsOnEdges[make_pair(min(id3, id1), max(id3, id1))] = intersection;
				}
				resPoint.emplace_back(make_pair(min(id3, id1), max(id3, id1)));
				flags[1] = true;
			}

			// 分情况讨论
			//  等值线过两条边
			if (resPoint.size() == 2)
			{
				fromOneEdgePoint2Neighbors[resPoint[0]].insert(resPoint[1]);
				fromOneEdgePoint2Neighbors[resPoint[1]].insert(resPoint[0]);
			}
			// 等值线过一条边与一个顶点
			else if (resPoint.size() == 1)
			{
				if (flags[0] && scalar1 == val) // 过点p1和边[p2,p3]
				{
					fromOneEdgePoint2Neighbors[make_pair(id1, id1)].insert(resPoint[0]);
					fromOneEdgePoint2Neighbors[resPoint[0]].insert(make_pair(id1, id1));
				}
				else if (flags[1] && scalar2 == val) // 过点p2和边[p3,p1]
				{
					fromOneEdgePoint2Neighbors[make_pair(id2, id2)].insert(resPoint[0]);
					fromOneEdgePoint2Neighbors[resPoint[0]].insert(make_pair(id2, id2));
				}
				else if (flags[2] && scalar3 == val) // 过点p3和边[p1,p2]
				{
					fromOneEdgePoint2Neighbors[make_pair(id3, id3)].insert(resPoint[0]);
					fromOneEdgePoint2Neighbors[resPoint[0]].insert(make_pair(id3, id3));
				}
			}
		}
	}

	vector<vector<V3d>>loops;
	while (!fromOneEdgePoint2Neighbors.empty())
	{
		vector<V3d>loop;
		auto firstPoint = fromOneEdgePoint2Neighbors.begin()->first;  // firstPoint为pair值
		auto prePoint = firstPoint;
		auto nxtPoint = *fromOneEdgePoint2Neighbors[firstPoint].begin();

		// 去掉相邻两点的连接关系, 并如果点的连接关系全部删除，则把该元素删除
		fromOneEdgePoint2Neighbors[prePoint].erase(nxtPoint);
		if (fromOneEdgePoint2Neighbors[prePoint].empty())
			fromOneEdgePoint2Neighbors.erase(prePoint);
		fromOneEdgePoint2Neighbors[nxtPoint].erase(prePoint);
		if (fromOneEdgePoint2Neighbors[nxtPoint].empty())
			fromOneEdgePoint2Neighbors.erase(nxtPoint);
		loop.emplace_back(pointsOnEdges[prePoint]);
		while (nxtPoint != firstPoint)
		{
			prePoint = nxtPoint;
			nxtPoint = *fromOneEdgePoint2Neighbors[prePoint].begin();

			// 去掉相邻两点的连接关系, 并如果点的连接关系全部删除，则把该元素删除
			fromOneEdgePoint2Neighbors[prePoint].erase(nxtPoint);
			if (fromOneEdgePoint2Neighbors[prePoint].empty())
				fromOneEdgePoint2Neighbors.erase(prePoint);
			fromOneEdgePoint2Neighbors[nxtPoint].erase(prePoint);
			if (fromOneEdgePoint2Neighbors[nxtPoint].empty())
				fromOneEdgePoint2Neighbors.erase(nxtPoint);
			loop.emplace_back(pointsOnEdges[prePoint]);
		}
		loops.emplace_back(loop);
	}
	return loops;
}

std::pair<BaseModel, BaseModel> BaseModel::splitModelByIsoline(const vector<double>& scalarField, const double& val)const
{
	vector<V3i> faceLess;
	vector<V3i> faceLarger;

	auto verts_copy = modelVerts;
	set<int>splitedFaces;               // 需要切分的面的id的集合

	for (int index = 0; index < modelFaces.size(); index++)
	{
		auto t = modelFaces[index];
		int id1 = t.x();
		int id2 = t.y();
		int id3 = t.z();
		auto p1 = modelVerts[id1];
		auto p2 = modelVerts[id2];
		auto p3 = modelVerts[id3];
		double scalar1 = scalarField[id1];
		double scalar2 = scalarField[id2];
		double scalar3 = scalarField[id3];

		// 等值线与某条边重合
		if (scalar1 == val && scalar2 == val)
		{
			if (scalar3 > val)
				faceLarger.emplace_back(t);
			else if (scalar3 < val)
				faceLess.emplace_back(t);
		}
		else if (scalar2 == val && scalar3 == val)
		{
			if (scalar1 > val)
				faceLarger.emplace_back(t);
			else if (scalar1 < val)
				faceLess.emplace_back(t);
		}
		else if (scalar3 == val && scalar1 == val)
		{
			if (scalar2 > val)
				faceLarger.emplace_back(t);
			else if (scalar2 < val)
				faceLess.emplace_back(t);
		}
		else
		{
			int count(0);
			bool flags[3] = { false };       // 用于判断在哪条边上有交
			map<PII, int>fromTwoIDs2IDofNewVert;   // 根据插入点所在边<min, max>，插入新的点，并且生成新的id
			// 等值线穿过边
			if ((scalar1 > val) != (scalar2 > val))
			{
				count++;
				double proportion = (scalar1 - val) / (scalar1 - scalar2);
				V3d intersection = proportion * p2 + (1 - proportion) * p1;
				if (fromTwoIDs2IDofNewVert.find(make_pair(min(id1, id2), max(id1, id2))) == fromTwoIDs2IDofNewVert.end())
				{
					verts_copy.emplace_back(intersection);
					fromTwoIDs2IDofNewVert[make_pair(min(id1, id2), max(id1, id2))] = verts_copy.size() - 1;
				}
				flags[2] = true;
			}
			if ((scalar2 > val) != (scalar3 > val))
			{
				count++;
				double proportion = (scalar2 - val) / (scalar2 - scalar3);
				V3d intersection = proportion * p3 + (1 - proportion) * p2;
				if (fromTwoIDs2IDofNewVert.find(make_pair(min(id2, id3), max(id2, id3))) == fromTwoIDs2IDofNewVert.end())
				{
					verts_copy.emplace_back(intersection);
					fromTwoIDs2IDofNewVert[make_pair(min(id2, id3), max(id2, id3))] = verts_copy.size() - 1;
				}
				flags[0] = true;
			}
			if ((scalar3 > val) != (scalar1 > val))
			{
				count++;
				double proportion = (scalar3 - val) / (scalar3 - scalar1);
				V3d intersection = proportion * p1 + (1 - proportion) * p3;
				if (fromTwoIDs2IDofNewVert.find(make_pair(min(id3, id1), max(id3, id1))) == fromTwoIDs2IDofNewVert.end())
				{
					verts_copy.emplace_back(intersection);
					fromTwoIDs2IDofNewVert[make_pair(min(id3, id1), max(id3, id1))] = verts_copy.size() - 1;
				}
				flags[1] = true;
			}

			if (count != 1 && count != 2)
				continue;

			splitedFaces.insert(index);
			//  等值线过两条边
			if (count == 2)
			{
				//splitedFaces.insert(index);

				if (flags[0] && flags[1])
				{
					// 过边p2p3和p1p3
					auto pre = fromTwoIDs2IDofNewVert[make_pair(min(id2, id3), max(id2, id3))];
					auto nxt = fromTwoIDs2IDofNewVert[make_pair(min(id3, id1), max(id3, id1))];

					if (scalar3 < val)
					{
						faceLess.emplace_back(Eigen::Vector3i(pre, id3, nxt));
						faceLarger.emplace_back(Eigen::Vector3i(id1, pre, nxt));
						faceLarger.emplace_back(Eigen::Vector3i(id1, id2, pre));
					}
					else
					{
						faceLarger.emplace_back(Eigen::Vector3i(pre, id3, nxt));
						faceLess.emplace_back(Eigen::Vector3i(id1, pre, nxt));
						faceLess.emplace_back(Eigen::Vector3i(id1, id2, pre));
					}
				}
				else if (flags[1] && flags[2])
				{
					// 过边p2p3和p1p3
					auto pre = fromTwoIDs2IDofNewVert[make_pair(min(id3, id1), max(id3, id1))];
					auto nxt = fromTwoIDs2IDofNewVert[make_pair(min(id1, id2), max(id1, id2))];

					if (scalar1 < val)
					{
						faceLess.emplace_back(Eigen::Vector3i(pre, id1, nxt));
						faceLarger.emplace_back(Eigen::Vector3i(id2, pre, nxt));
						faceLarger.emplace_back(Eigen::Vector3i(id2, id3, pre));
					}
					else
					{
						faceLarger.emplace_back(Eigen::Vector3i(pre, id1, nxt));
						faceLess.emplace_back(Eigen::Vector3i(id2, pre, nxt));
						faceLess.emplace_back(Eigen::Vector3i(id2, id3, pre));
					}
				}
				else if (flags[2] && flags[0])
				{
					// 过边p2p3和p1p3
					auto pre = fromTwoIDs2IDofNewVert[make_pair(min(id1, id2), max(id1, id2))];
					auto nxt = fromTwoIDs2IDofNewVert[make_pair(min(id2, id3), max(id2, id3))];

					if (scalar2 < val)
					{
						faceLess.emplace_back(Eigen::Vector3i(pre, id2, nxt));
						faceLarger.emplace_back(Eigen::Vector3i(id3, pre, nxt));
						faceLarger.emplace_back(Eigen::Vector3i(id3, id1, pre));
					}
					else
					{
						faceLarger.emplace_back(Eigen::Vector3i(pre, id2, nxt));
						faceLess.emplace_back(Eigen::Vector3i(id3, pre, nxt));
						faceLess.emplace_back(Eigen::Vector3i(id3, id1, pre));
					}
				}
			}
			// 等值线过一条边与一个顶点
			if (count == 1)
			{
				if (flags[0] && scalar1 == val) // 过点p1和边[p2,p3]
				{
					auto new_id = fromTwoIDs2IDofNewVert[make_pair(min(id2, id3), max(id2, id3))];
					if (scalar2 < val)
					{
						faceLess.emplace_back(Eigen::Vector3i(id1, id2, new_id));
						faceLarger.emplace_back(Eigen::Vector3i(id1, new_id, id3));
					}
					else
					{
						faceLarger.emplace_back(Eigen::Vector3i(id1, id2, new_id));
						faceLess.emplace_back(Eigen::Vector3i(id1, new_id, id3));
					}
				}
				else if (flags[1] && scalar2 == val) // 过点p2和边[p3,p1]
				{
					auto new_id = fromTwoIDs2IDofNewVert[make_pair(min(id3, id1), max(id3, id1))];
					if (scalar1 < val)
					{
						faceLess.emplace_back(Eigen::Vector3i(id1, id2, new_id));
						faceLarger.emplace_back(Eigen::Vector3i(id3, new_id, id2));
					}
					else
					{
						faceLarger.emplace_back(Eigen::Vector3i(id1, id2, new_id));
						faceLess.emplace_back(Eigen::Vector3i(id3, new_id, id2));
					}
				}
				else if (flags[2] && scalar3 == val) // 过点p3和边[p1,p2]
				{
					auto new_id = fromTwoIDs2IDofNewVert[make_pair(min(id1, id2), max(id1, id2))];
					if (scalar1 < val)
					{
						faceLess.emplace_back(Eigen::Vector3i(id3, id1, new_id));
						faceLarger.emplace_back(Eigen::Vector3i(id3, new_id, id2));
					}
					else
					{
						faceLarger.emplace_back(Eigen::Vector3i(id3, id1, new_id));
						faceLess.emplace_back(Eigen::Vector3i(id3, new_id, id2));
					}
				}
			}
		}
	}

	for (int index = 0; index < modelFaces.size(); index++)
	{
		if (splitedFaces.find(index) == splitedFaces.end())
		{
			if (scalarField[modelFaces[index].x()] > val)
			{
				faceLarger.emplace_back(modelFaces[index]);
			}
			else
				faceLess.emplace_back(modelFaces[index]);
		}

	}

	// 更改点的序号
	vector<V3d>vertsLarger; // >val模型
	map<int, int>fromOldID2NewIDforLarger;
	for (int index = 0; index < faceLarger.size(); index++)
	{
		auto t = faceLarger[index];
		int oldIDs[] = { t.x(),t.y(),t.z() };
		for (int i = 0; i < 3; i++)
		{
			int id = oldIDs[i];
			if (fromOldID2NewIDforLarger.find(id) == fromOldID2NewIDforLarger.end())
			{
				vertsLarger.emplace_back(verts_copy[id]);
				fromOldID2NewIDforLarger[id] = vertsLarger.size() - 1;
			}
		}
		faceLarger[index] = Eigen::Vector3i(fromOldID2NewIDforLarger[oldIDs[0]], fromOldID2NewIDforLarger[oldIDs[1]], fromOldID2NewIDforLarger[oldIDs[2]]);
	}

	vector<V3d>vertsLess; // <val模型
	map<int, int>fromOldID2NewIDforLess;
	for (int index = 0; index < faceLess.size(); index++)
	{
		auto t = faceLess[index];
		int oldIDs[] = { t.x(),t.y(),t.z() };
		for (int i = 0; i < 3; i++)
		{
			int id = oldIDs[i];
			if (fromOldID2NewIDforLess.find(id) == fromOldID2NewIDforLess.end())
			{
				vertsLess.emplace_back(verts_copy[id]);
				fromOldID2NewIDforLess[id] = vertsLess.size() - 1;
			}
		}
		faceLess[index] = Eigen::Vector3i(fromOldID2NewIDforLess[oldIDs[0]], fromOldID2NewIDforLess[oldIDs[1]], fromOldID2NewIDforLess[oldIDs[2]]);
	}

	return make_pair(BaseModel(vertsLarger, faceLarger), BaseModel(vertsLess, faceLess));
}

MXd BaseModel::getVertices() const
{
	return m_V;
}

MXi BaseModel::getFaces() const
{
	return m_F;
}

MXd BaseModel::getClosestPoint(const MXd& queryPointMat) const
{
	Eigen::VectorXd sqrD;
	Eigen::VectorXi I;
	Eigen::MatrixXd C;

	// the output sqrD contains the (unsigned) squared distance from each point in P 
	// to its closest point given in C which lies on the element in F given by I
	aabbTree.squared_distance(m_V, m_F, queryPointMat, sqrD, I, C);

	return C;
}

inline MXd BaseModel::getSurfacePointNormal(const MXd& queryPointMat)
{
	const size_t numPoint = queryPointMat.rows();

	Eigen::VectorXd sqrD;
	Eigen::VectorXi I;
	Eigen::MatrixXd C;
	getClosestPoint(queryPointMat);

	Eigen::MatrixXd resNormal(numPoint, 3);
	for (int i = 0; i < numPoint; ++i)
	{
		Eigen::Vector3d normal = m_FN.row(I(i)).normalized();
		resNormal.row(i) = normal;
	}
	return resNormal;
}

//////////////////////
//    I/O: Model    //
//////////////////////
void BaseModel::readFile(const string& filename)
{
	if (!igl::read_triangle_mesh(filename, m_V, m_F))
	{
		fprintf(stderr, "[I/O] Error: File %s could not open!", filename.c_str());
		exit(EXIT_FAILURE);
	}
	modelName = getFileName(DELIMITER, filename);
}

void BaseModel::setModelAttributeVector()
{
	for (int i = 0; i < m_V.rows(); i++) modelVerts.emplace_back(m_V.row(i));
	for (int i = 0; i < m_F.rows(); i++)
	{
		modelFaces.emplace_back(m_F.row(i));

		modelTris.emplace_back(Triangle<Eigen::Vector3d>(modelVerts[(m_F.row(i))[0]],
			modelVerts[(m_F.row(i))[1]],
			modelVerts[(m_F.row(i))[2]]
		));
	}

	nModelVerts = modelVerts.size(), nModelTris = modelFaces.size();

	igl::per_vertex_normals(m_V, m_F, m_VN);
	igl::per_face_normals_stable(m_V, m_F, m_FN);

	aabbTree.init(m_V, m_F);
}

void BaseModel::readOffFile(const string& filename)
{
	ifstream in(filename);
	if (!in.is_open())
	{
		cout << "Unable to open the file!" << endl;
	}
	string line;
	in >> line;  // off字符串
	int VertexNum, FaceNum, EdgeNum;
	in >> VertexNum >> FaceNum >> EdgeNum;   // 读取顶点、面、边的数量

	for (size_t i = 0; i < VertexNum; i++)
	{
		double x, y, z;
		in >> x >> y >> z;
		modelVerts.emplace_back(V3d(x, y, z));
		//cout << x << " " << y << " " << z << endl;
	}

	for (size_t i = 0; i < FaceNum; i++)
	{

		int EdgeNumOfFace;   // 网格面片形状
		in >> EdgeNumOfFace;

		vector<int>indices;
		for (int j = 0; j < EdgeNumOfFace; j++)
		{

			int index;
			in >> index;
			indices.emplace_back(index);
		}
		for (int j = 1; j < EdgeNumOfFace - 1; j++)
		{
			modelFaces.emplace_back(V3i(indices[0], indices[j], indices[j + 1]));
			//cout << indices[0] <<" " << indices[j] << " " << indices[j + 1] << endl;
		}
	}
	in.close();
}

void BaseModel::readObjFile(const string& filename)
{
	ifstream in(filename);
	if (!in)
	{
		fprintf(stderr, "Unable to open the File: %s !\n");
		exit(-1);
	}

	char buf[1024];
	while (in.getline(buf, sizeof(buf)))   // 读入文件行于buf
	{
		std::stringstream line(buf);         // 将line改为字符串
		string word;

		line >> word;                  // word为第一个字符串，以空格为标识
		if (word == "v")
		{
			double x, y, z;
			line >> x >> y >> z;
			modelVerts.emplace_back(V3d(x, y, z));

			//cout << "v " << x << y << z << endl;
		}

		else if (word == "f")
		{
			vector<int>indices;
			int index;

			while (line >> index)
			{
				indices.emplace_back(index);
			}
			for (int i = 1; i < indices.size() - 1; i++)
			{
				modelFaces.emplace_back(Eigen::Vector3i(indices[0] - 1, indices[i] - 1, indices[i + 1] - 1));
			}
		}
	}
	in.close();
}

void BaseModel::writeObjFile(const string& filename) const
{
	std::ofstream out(filename);
	out << "# Vertices: " << modelVerts.size() << "\tFaces: " << modelFaces.size() << endl;
	for (auto v : modelVerts)
	{
		out << "v " << v.x() << " " << v.y() << " " << v.z() << " " << endl;
	}
	for (auto f : modelFaces)
	{
		out << "f " << (f + V3i(1, 1, 1)).transpose() << endl;
	}
	out.close();
}

void BaseModel::writeObjFile(const string& filename, const vector<V3d>& V, const vector<V3i>& F) const
{
	std::ofstream out(filename);
	out << "# Vertices: " << modelVerts.size() << "\tFaces: " << modelFaces.size() << endl;
	for (size_t i = 0; i < V.size(); i++)
	{
		V3d v = V[i];
		out << "v " << std::fixed << std::setprecision(5) << v.x() << " " << v.y() << " " << v.z() << " " << endl;
	}
	for (size_t i = 0; i < F.size(); i++)
	{
		V3i f = F[i];
		out << "f " << (f + V3i(1, 1, 1)).transpose() << endl;
	}
	out.close();
}

void BaseModel::saveVertices(const string& filename, const vector<V3d>& verts)
{
	std::ofstream out(filename);
	out << "# " << verts.size() << " vertices " << std::endl;

	for (const V3d& v : verts)
	{
		out << "v " << std::setiosflags(std::ios::fixed) << std::setprecision(10) << v.x() << " " << v.y() << " " << v.z() << std::endl;
	}
	out.close();
}

void BaseModel::saveIsoline(const string& filename, const vector<vector<V3d>>& isoline)const
{
	std::ofstream out(filename);
	out << "g 3d_lines" << endl;
	int cnt(0);
	for (auto loop : isoline)
	{
		/*for (auto v : loop)
		{
			out << "v " << v.transpose() << endl;
		}*/
		for (int i = 0; i < loop.size(); i++)
		{
			out << "v " << loop[i].x() << " " << loop[i].y() << " " << loop[i].z() << endl;
		}
		//out << "l ";
		for (int i = 1; i <= loop.size() - 1; i++)
		{
			out << "l " << i + cnt << " " << i + 1 + cnt << endl;;
		}
		out << "l " << loop.size() << " " << 1 << endl;
		//out << 1 + cnt << endl;
		cnt += loop.size();
	}

	out.close();
}

void BaseModel::writeTexturedObjFile(const string& filename, const vector<PDD>& uvs)const
{
	std::ofstream out(filename);
	out << "# Vertices: " << modelVerts.size() << "\tFaces: " << modelFaces.size() << endl;
	out << "mtllib defaultmaterial.mtl" << endl;
	out << "usemtl mydefault" << endl;
	for (auto v : modelVerts)
	{
		out << "v " << v.x() << " " << v.y() << " " << v.z() << " " << endl;
	}
	for (auto uv : uvs)
	{
		out << "vt " << uv.first << " " << uv.second << " " << endl;
	}
	for (auto f : modelFaces)
	{
		auto ids = (f + Eigen::Vector3i(1, 1, 1)).transpose();
		out << "f " << ids.x() << "/" << ids.x() << " " << ids.y() << "/" << ids.y() << " " << ids.z() << "/" << ids.z() << endl;
	}
	out.close();
}

void BaseModel::writeTexturedObjFile(const string& filename, const VXd& uvs)const
{
	std::ofstream out(filename);
	out << "# Vertices: " << modelVerts.size() << "\tFaces: " << modelFaces.size() << endl;
	/*out << "mtllib defaultmaterial.mtl" << endl;
	out << "usemtl mydefault" << endl;*/
	for (auto v : modelVerts)
	{
		out << "v " << v.x() << " " << v.y() << " " << v.z() << " " << endl;
	}
	// 纹理坐标需介于0-1之间
	//auto maxU = *max_element(uvs.begin(), uvs.end());
	double maxU = uvs.maxCoeff();
	double minU = uvs.minCoeff();
	for (auto u : uvs)
	{
		out << "vt " << (u - minU) / (maxU - minU) << " " << 0 << " " << endl;
	}
	int i = 0;
	for (auto f : modelFaces)
	{
		V3i ids = f + V3i(1, 1, 1);
		out << "f " << ids.x() << "/" << ids.x() << " " << ids.y() << "/" << ids.y() << " " << ids.z() << "/" << ids.z() << endl;
	}
	out.close();
}