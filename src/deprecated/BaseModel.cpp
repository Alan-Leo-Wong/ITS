#include "BaseModel.hpp"
#include "utils/IO.hpp"
#include "utils/Common.hpp"
#include "utils/String.hpp"
#include "CUDACompute.hpp"
#include <sstream>
#include <iomanip>
#include <igl/writeOBJ.h>
#include <igl/read_triangle_mesh.h>
#include <igl/per_vertex_normals.h>
#include <igl/point_mesh_squared_distance.h>

NAMESPACE_BEGIN(ITS)
    namespace core {

        //////////////////////
        //   Constructors   //
        //////////////////////
        BaseModel::BaseModel(std::vector<Vector3d> verts, std::vector<Vector3i> faces) :
                modelVerts(std::move(verts)), modelFaces(std::move(faces)),
                is2UnitCube(false), scaleFactor(1.0) {
            setUniformBoundingBox();
            setTriAttributes();
        }

        BaseModel::BaseModel(const std::string &filename) {
            readFile(filename);
            setModelAttributeVector();
            setUniformBoundingBox();
            setTriAttributes();
        }

        BaseModel::BaseModel(const std::string &filename, bool _is2UnitCube, double _scaleFactor)
                : is2UnitCube(_is2UnitCube), scaleFactor(_scaleFactor) {
            readFile(filename);
            if (_is2UnitCube) {
                uniformDir = "uniform";
                model2UnitCube();
            }
            setModelAttributeVector();
            setUniformBoundingBox();
            setTriAttributes();
        }

        BaseModel::BaseModel(const std::string &filename, bool _is2UnitCube, double _scaleFactor,
                  bool _isAddNoise, double noisePercentage)
                : is2UnitCube(_is2UnitCube), scaleFactor(_scaleFactor) {
            readFile(filename);
            if (_is2UnitCube) {
                uniformDir = "uniform";
                model2UnitCube();
            }
            if (_isAddNoise) {
                printf("Add %lf%% noise on model...\n", noisePercentage * 100.0);
                noiseDir = (std::string) "noise_" + std::to_string(noisePercentage * 100.0);
                addNoise(noisePercentage);
            }
            setModelAttributeVector();
            setUniformBoundingBox();
            setTriAttributes();
        }

        //////////////////////
        //   Model  Utils   //
        //////////////////////
        void BaseModel::setBoundingBox(const float &scaleSize) {
            Vector3d minV = m_V.colwise().minCoeff();
            Vector3d maxV = m_V.colwise().maxCoeff();

            modelBoundingBox = AABox<Vector3d>(minV, maxV);
        }

        void BaseModel::setUniformBoundingBox() {
            Vector3d minV = m_V.colwise().minCoeff();
            Vector3d maxV = m_V.colwise().maxCoeff();

            modelBoundingBox = AABox<Vector3d>(minV, maxV); // initialize answer
            Eigen::Vector3d lengths = maxV - minV; // check length of given bbox in every direction
            float max_length = fmaxf(lengths.x(), fmaxf(lengths.y(), lengths.z())); // find max length
            for (unsigned int i = 0; i < 3; i++) { // for every direction (X,Y,Z)
                if (max_length == lengths[i]) {
                    continue;
                } else {
                    float delta = max_length -
                                  lengths[i]; // compute difference between largest length and current (X,Y or Z) length
                    modelBoundingBox.boxOrigin[i] =
                            minV[i] - (delta / 2.0f); // pad with half the difference before current min
                    modelBoundingBox.boxEnd[i] =
                            maxV[i] + (delta / 2.0f); // pad with half the difference behind current max
                }
            }

            // Next snippet adresses the problem reported here: https://github.com/Forceflow/cuda_voxelizer/issues/7
            // Suspected cause: If a triangle is axis-aligned and lies perfectly on a voxel edge, it sometimes gets counted / not counted
            // Probably due to a numerical instability (division by zero?)
            // Ugly fix: we pad the bounding box on all sides by 1/10001th of its total length, bringing all triangles ever so slightly off-grid
            Eigen::Vector3d epsilon = (modelBoundingBox.boxEnd - modelBoundingBox.boxOrigin) / 11; // ֮ǰ��10001
            modelBoundingBox.boxOrigin -= epsilon;
            modelBoundingBox.boxEnd += epsilon;
            modelBoundingBox.boxWidth = modelBoundingBox.boxEnd - modelBoundingBox.boxOrigin;
        }

        // ����to unit cube
        Eigen::Matrix4d BaseModel::calcUnitCubeTransformMatrix() {
            Eigen::RowVector3d boxMin = m_V.colwise().minCoeff();
            Eigen::RowVector3d boxMax = m_V.colwise().maxCoeff();

            // Get the target solveRes (along the largest dimension)
            double scale = boxMax[0] - boxMin[0];
            double minScale = scale;
            for (int d = 1; d < 3; d++) {
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
            for (int i = 0; i < 3; i++) {
                zoomMatrix(i, i) = 1. / scale;
                transMatrix(3, i) = -center[i];
            }
            return zoomMatrix * transMatrix;
        }

        void BaseModel::model2UnitCube() {
            auto transMat = calcUnitCubeTransformMatrix();
            for (int i = 0; i < m_V.rows(); ++i) {
                m_V.row(i) += transMat.block(3, 0, 1, 3);
                m_V.row(i) = m_V.row(i) * transMat.block(0, 0, 3, 3);
            }
        }

        void BaseModel::unitCube2Model() {
            auto transMat = calcUnitCubeTransformMatrix();
            Eigen::Matrix3d inverseTrans = transMat.block(0, 0, 3, 3).inverse();
            for (int i = 0; i < m_V.rows(); ++i) {
                m_V.row(i) = m_V.row(i) * inverseTrans;
                m_V.row(i) -= transMat.block(3, 0, 1, 3);
            }
        }

        // ��������ģ��
        Eigen::Matrix3d BaseModel::calcScaleMatrix() {
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

        void BaseModel::addNoise(const double &noisePercentage, const double &min_val, const double &max_val) {
            // ������Ҫ�Ŷ��Ľڵ�����
            int numNodes = m_V.rows();
            int numNoisyNodes = static_cast<int>(noisePercentage * numNodes);

            // ���������������
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dis(min_val, max_val); // �����Ŷ���ΧΪ [-0.1, 0.1]

            // ����Ŷ��ڵ�λ��
            for (int i = 0; i < numNoisyNodes; i++) {
                // ���ѡ��һ���ڵ�����
                int nodeIndex = std::rand() % numNodes;

                // �ڽڵ�λ�����������Ŷ�
                m_V.row(nodeIndex) += dis(gen) * Eigen::RowVector3d::Random();
            }
        }

        void BaseModel::zoomModel() {
            Eigen::Matrix3d zoomMat = calcScaleMatrix();
            m_V = m_V * zoomMat;
        }

        // ����һ���transform��������ģ�ͣ��ٽ��µ����ĵ���������ǰģ�͵����ĵ�
        Eigen::Matrix4d BaseModel::calcTransformMatrix(const float &_scaleFactor) {
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
            for (int i = 0; i < 3; i++) {
                zoomMatrix(i, i) = scaleFactor;
                transMatrix(3, i) = center[i] * (scaleFactor - 1);
            }
            return zoomMatrix * transMatrix;
        }

        void BaseModel::transformModel(const float &_scaleFactor) {
            Eigen::Matrix4d transMat = calcTransformMatrix(_scaleFactor);
            for (int i = 0; i < m_V.rows(); ++i) {
                m_V.row(i) += transMat.block(3, 0, 1, 3);
                m_V.row(i) = m_V.row(i) * transMat.block(0, 0, 3, 3);
            }
        }

        Eigen::MatrixXd
        BaseModel::generateGaussianRandomPoints(const size_t &numPoints, const float &_scaleFactor, const float &dis) {
            Eigen::MatrixXd M;
            const Eigen::RowVector3d min_area = modelBoundingBox.boxOrigin;
            const Eigen::RowVector3d max_area = modelBoundingBox.boxEnd;
            donut::getGaussianRandomMatrix<double>(min_area, max_area, numPoints, _scaleFactor, dis, M);
            return M;
        }

        Eigen::MatrixXd
        BaseModel::generateUniformRandomPoints(const size_t &numPoints, const float &_scaleFactor, const float &dis) {
            Eigen::MatrixXd M;
            const Eigen::RowVector3d min_area = modelBoundingBox.boxOrigin;
            const Eigen::RowVector3d max_area = modelBoundingBox.boxEnd;
            donut::getUniformRandomMatrix<double>(min_area, max_area, numPoints, _scaleFactor, dis, M);
            return M;
        }

        Eigen::MatrixXd BaseModel::generateGaussianRandomPoints(const std::string &filename, const size_t &numPoints,
                                                                const float &_scaleFactor, const float &dis) {
            Eigen::MatrixXd M;
            const Eigen::RowVector3d min_area = modelBoundingBox.boxOrigin;
            const Eigen::RowVector3d max_area = modelBoundingBox.boxEnd;
            donut::getGaussianRandomMatrix<double>(min_area, max_area, numPoints, _scaleFactor, dis, M);

            str_util::checkDir(filename);
            std::ofstream out(filename, std::ofstream::out);
            if (!out) {
                fprintf(stderr, "[I/O] Error: File %s could not be opened!", filename.c_str());
                return M;
            }
            std::cout << "-- Save random points to " << std::quoted(filename) << std::endl;

            //std::cout << getFileExtension(filename) << std::endl;
            if (str_util::getFileExtension(filename) == ".obj")
                gvis::writePointCloud(M, out);
            else if (str_util::getFileExtension(filename) == ".xyz")
                gvis::writePointCloud_xyz(M, out);

            return M;
        }

        Eigen::MatrixXd BaseModel::generateUniformRandomPoints(const std::string &filename, const size_t &numPoints,
                                                               const float &_scaleFactor, const float &dis) {
            Eigen::MatrixXd M;
            const Eigen::RowVector3d min_area = modelBoundingBox.boxOrigin;
            const Eigen::RowVector3d max_area = modelBoundingBox.boxEnd;
            donut::getUniformRandomMatrix<double>(min_area, max_area, numPoints, _scaleFactor, dis, M);

            str_util::checkDir(filename);
            std::ofstream out(filename, std::ofstream::out);
            if (!out) {
                fprintf(stderr, "[I/O] Error: File %s could not be opened!", filename.c_str());
                return M;
            }
            std::cout << "-- Save random points to " << std::quoted(filename) << std::endl;

            //std::cout << getFileExtension(filename) << std::endl;
            if (str_util::getFileExtension(filename) == ".obj")
                gvis::writePointCloud(M, out);
            else if (str_util::getFileExtension(filename) == ".xyz")
                gvis::writePointCloud_xyz(M, out);

            return M;
        }

        std::vector<Vector3d>
        BaseModel::generateUniformRandomPoints(const std::string &filename, const size_t &numPoints,
                                               const double &_scaleFactor, const Vector3d &dis) {
            std::vector<Vector3d> randomPoints;
            donut::getUniformRandomMatrix<Vector3d>(modelBoundingBox, numPoints, _scaleFactor, dis, randomPoints);

            str_util::checkDir(filename);
            std::ofstream out(filename, std::ofstream::out);
            if (!out) {
                fprintf(stderr, "[I/O] Error: File %s could not be opened!", filename.c_str());
                return randomPoints;
            }
            std::cout << "-- Save random points to " << std::quoted(filename) << std::endl;

            //std::cout << getFileExtension(filename) << std::endl;
            if (str_util::getFileExtension(filename) == ".obj")
                gvis::writePointCloud(randomPoints, out);
            else if (str_util::getFileExtension(filename) == ".xyz")
                gvis::writePointCloud_xyz(randomPoints, out);

            return randomPoints;
        }

        std::vector<Vector3d>
        BaseModel::generateGaussianRandomPoints(const std::string &filename, const size_t &numPoints,
                                                const double &_scaleFactor, const Vector3d &dis) {
            std::vector<Vector3d> randomPoints;
            donut::getGaussianRandomMatrix<Vector3d>(modelBoundingBox, numPoints, _scaleFactor, dis, randomPoints);

            str_util::checkDir(filename);
            std::ofstream out(filename, std::ofstream::out);
            if (!out) {
                fprintf(stderr, "[I/O] Error: File %s could not be opened!", filename.c_str());
                return randomPoints;
            }
            std::cout << "-- Save random points to " << std::quoted(filename) << std::endl;

            //std::cout << getFileExtension(filename) << std::endl;
            if (str_util::getFileExtension(filename) == ".obj")
                gvis::writePointCloud(randomPoints, out);
            else if (str_util::getFileExtension(filename) == ".xyz")
                gvis::writePointCloud_xyz(randomPoints, out);

            return randomPoints;
        }

        void BaseModel::setTriAttributes() {
            cuAcc::launch_modelTriAttributeKernel(nModelTris, modelTris);
        }

        std::vector<Vector2i> BaseModel::extractEdges() {
            std::cout << "Extracting edges from " << std::quoted(modelName) << std::endl;

            std::vector<Vector2i> edges;
            std::set<PII> uset;

            for (Vector3i f: modelFaces) {
                int maxF = f.maxCoeff();
                int minF = f.minCoeff();
                int middleF = f.sum() - maxF - minF;

                uset.insert(std::make_pair(minF, middleF));
                uset.insert(std::make_pair(middleF, maxF));
                uset.insert(std::make_pair(minF, maxF));
            }
            for (PII it: uset)
                edges.emplace_back(Vector2i(it.first, it.second));

            std::cout << "-- Number of " << modelName << "'edges: " << edges.size() << std::endl;
            return edges;
        }

        std::vector<std::vector<Vector3d>>
        BaseModel::extractIsoline(const std::vector<double> &scalarField, const double &val) const {
            using namespace std;
            map<PII, Vector3d> pointsOnEdges;  // ���ݱߵ�pairֵ���������λ��
            map<PII, std::set<PII>> fromOneEdgePoint2Neighbors;   // keyֵΪ���ϵ㣬value��ʾ���ڵ�������
            for (Vector3i t: modelFaces) {
                int id1 = t.x();
                int id2 = t.y();
                int id3 = t.z();
                Vector3d p1 = modelVerts[id1];
                Vector3d p2 = modelVerts[id2];
                Vector3d p3 = modelVerts[id3];
                double scalar1 = scalarField[id1];
                double scalar2 = scalarField[id2];
                double scalar3 = scalarField[id3];

                // ��ֵ����ĳ�����غ�
                if (scalar1 == val && scalar2 == val) {
                    if (pointsOnEdges.find(make_pair(id1, id1)) == pointsOnEdges.end()) {
                        pointsOnEdges[make_pair(id1, id1)] = p1;
                    }
                    fromOneEdgePoint2Neighbors[make_pair(id1, id1)].insert(make_pair(id2, id2));
                    if (pointsOnEdges.find(make_pair(id2, id2)) == pointsOnEdges.end()) {
                        pointsOnEdges[make_pair(id2, id2)] = p2;
                    }
                    fromOneEdgePoint2Neighbors[make_pair(id2, id2)].insert(make_pair(id1, id1));
                } else if (scalar2 == val && scalar3 == val) {
                    if (pointsOnEdges.find(make_pair(id2, id2)) == pointsOnEdges.end()) {
                        pointsOnEdges[make_pair(id2, id2)] = p2;
                    }
                    fromOneEdgePoint2Neighbors[make_pair(id2, id2)].insert(make_pair(id3, id3));
                    if (pointsOnEdges.find(make_pair(id3, id3)) == pointsOnEdges.end()) {
                        pointsOnEdges[make_pair(id3, id3)] = p3;
                    }
                    fromOneEdgePoint2Neighbors[make_pair(id3, id3)].insert(make_pair(id2, id2));
                } else if (scalar3 == val && scalar1 == val) {
                    if (pointsOnEdges.find(make_pair(id3, id3)) == pointsOnEdges.end()) {
                        pointsOnEdges[make_pair(id3, id3)] = p3;
                    }
                    fromOneEdgePoint2Neighbors[make_pair(id3, id3)].insert(make_pair(id1, id1));
                    if (pointsOnEdges.find(make_pair(id1, id1)) == pointsOnEdges.end()) {
                        pointsOnEdges[make_pair(id1, id1)] = p1;
                    }
                    fromOneEdgePoint2Neighbors[make_pair(id1, id1)].insert(make_pair(id3, id3));
                } else {
                    std::vector<PII> resPoint;   // �洢����ĳ���ߵĽ������ڱߵ����˶˵�id
                    bool flags[3] = {false};       // �����ж������������н�
                    // ��ֵ�ߴ�����
                    if ((scalar1 > val) != (scalar2 > val)) {
                        if (fromOneEdgePoint2Neighbors.find(make_pair(min(id1, id2), max(id1, id2))) ==
                            fromOneEdgePoint2Neighbors.end()) {
                            double proportion = (scalar1 - val) / (scalar1 - scalar2);
                            Vector3d intersection = proportion * p2 + (1 - proportion) * p1;
                            pointsOnEdges[make_pair(min(id1, id2), max(id1, id2))] = intersection;
                        }
                        resPoint.emplace_back(make_pair(min(id1, id2), max(id1, id2)));
                        flags[2] = true;
                    }
                    if ((scalar2 > val) != (scalar3 > val)) {
                        if (fromOneEdgePoint2Neighbors.find(make_pair(min(id2, id3), max(id2, id3))) ==
                            fromOneEdgePoint2Neighbors.end()) {
                            double proportion = (scalar2 - val) / (scalar2 - scalar3);
                            Vector3d intersection = proportion * p3 + (1 - proportion) * p2;
                            pointsOnEdges[make_pair(min(id2, id3), max(id2, id3))] = intersection;
                        }
                        resPoint.emplace_back(make_pair(min(id2, id3), max(id2, id3)));
                        flags[0] = true;
                    }
                    if ((scalar3 > val) != (scalar1 > val)) {
                        if (fromOneEdgePoint2Neighbors.find(make_pair(min(id3, id1), max(id3, id1))) ==
                            fromOneEdgePoint2Neighbors.end()) {
                            double proportion = (scalar3 - val) / (scalar3 - scalar1);
                            Vector3d intersection = proportion * p1 + (1 - proportion) * p3;
                            pointsOnEdges[make_pair(min(id3, id1), max(id3, id1))] = intersection;
                        }
                        resPoint.emplace_back(make_pair(min(id3, id1), max(id3, id1)));
                        flags[1] = true;
                    }

                    // ���������
                    //  ��ֵ�߹�������
                    if (resPoint.size() == 2) {
                        fromOneEdgePoint2Neighbors[resPoint[0]].insert(resPoint[1]);
                        fromOneEdgePoint2Neighbors[resPoint[1]].insert(resPoint[0]);
                    }
                        // ��ֵ�߹�һ������һ������
                    else if (resPoint.size() == 1) {
                        if (flags[0] && scalar1 == val) // ����p1�ͱ�[p2,p3]
                        {
                            fromOneEdgePoint2Neighbors[make_pair(id1, id1)].insert(resPoint[0]);
                            fromOneEdgePoint2Neighbors[resPoint[0]].insert(make_pair(id1, id1));
                        } else if (flags[1] && scalar2 == val) // ����p2�ͱ�[p3,p1]
                        {
                            fromOneEdgePoint2Neighbors[make_pair(id2, id2)].insert(resPoint[0]);
                            fromOneEdgePoint2Neighbors[resPoint[0]].insert(make_pair(id2, id2));
                        } else if (flags[2] && scalar3 == val) // ����p3�ͱ�[p1,p2]
                        {
                            fromOneEdgePoint2Neighbors[make_pair(id3, id3)].insert(resPoint[0]);
                            fromOneEdgePoint2Neighbors[resPoint[0]].insert(make_pair(id3, id3));
                        }
                    }
                }
            }

            std::vector<std::vector<Vector3d>> loops;
            while (!fromOneEdgePoint2Neighbors.empty()) {
                std::vector<Vector3d> loop;
                auto firstPoint = fromOneEdgePoint2Neighbors.begin()->first;  // firstPointΪpairֵ
                auto prePoint = firstPoint;
                auto nxtPoint = *fromOneEdgePoint2Neighbors[firstPoint].begin();

                // ȥ��������������ӹ�ϵ, �����������ӹ�ϵȫ��ɾ������Ѹ�Ԫ��ɾ��
                fromOneEdgePoint2Neighbors[prePoint].erase(nxtPoint);
                if (fromOneEdgePoint2Neighbors[prePoint].empty())
                    fromOneEdgePoint2Neighbors.erase(prePoint);
                fromOneEdgePoint2Neighbors[nxtPoint].erase(prePoint);
                if (fromOneEdgePoint2Neighbors[nxtPoint].empty())
                    fromOneEdgePoint2Neighbors.erase(nxtPoint);
                loop.emplace_back(pointsOnEdges[prePoint]);
                while (nxtPoint != firstPoint) {
                    prePoint = nxtPoint;
                    nxtPoint = *fromOneEdgePoint2Neighbors[prePoint].begin();

                    // ȥ��������������ӹ�ϵ, �����������ӹ�ϵȫ��ɾ������Ѹ�Ԫ��ɾ��
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

        std::pair<BaseModel, BaseModel>
        BaseModel::splitModelByIsoline(const std::vector<double> &scalarField, const double &val) const {
            using namespace std;

            std::vector<Vector3i> faceLess;
            std::vector<Vector3i> faceLarger;

            auto verts_copy = modelVerts;
            set<int> splitedFaces;               // ��Ҫ�зֵ����id�ļ���

            for (int index = 0; index < modelFaces.size(); index++) {
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

                // ��ֵ����ĳ�����غ�
                if (scalar1 == val && scalar2 == val) {
                    if (scalar3 > val)
                        faceLarger.emplace_back(t);
                    else if (scalar3 < val)
                        faceLess.emplace_back(t);
                } else if (scalar2 == val && scalar3 == val) {
                    if (scalar1 > val)
                        faceLarger.emplace_back(t);
                    else if (scalar1 < val)
                        faceLess.emplace_back(t);
                } else if (scalar3 == val && scalar1 == val) {
                    if (scalar2 > val)
                        faceLarger.emplace_back(t);
                    else if (scalar2 < val)
                        faceLess.emplace_back(t);
                } else {
                    int count(0);
                    bool flags[3] = {false};       // �����ж������������н�
                    map<PII, int> fromTwoIDs2IDofNewVert;   // ���ݲ�������ڱ�<min, max>�������µĵ㣬���������µ�id
                    // ��ֵ�ߴ�����
                    if ((scalar1 > val) != (scalar2 > val)) {
                        count++;
                        double proportion = (scalar1 - val) / (scalar1 - scalar2);
                        Vector3d intersection = proportion * p2 + (1 - proportion) * p1;
                        if (fromTwoIDs2IDofNewVert.find(make_pair(min(id1, id2), max(id1, id2))) ==
                            fromTwoIDs2IDofNewVert.end()) {
                            verts_copy.emplace_back(intersection);
                            fromTwoIDs2IDofNewVert[make_pair(min(id1, id2), max(id1, id2))] = verts_copy.size() - 1;
                        }
                        flags[2] = true;
                    }
                    if ((scalar2 > val) != (scalar3 > val)) {
                        count++;
                        double proportion = (scalar2 - val) / (scalar2 - scalar3);
                        Vector3d intersection = proportion * p3 + (1 - proportion) * p2;
                        if (fromTwoIDs2IDofNewVert.find(make_pair(min(id2, id3), max(id2, id3))) ==
                            fromTwoIDs2IDofNewVert.end()) {
                            verts_copy.emplace_back(intersection);
                            fromTwoIDs2IDofNewVert[make_pair(min(id2, id3), max(id2, id3))] = verts_copy.size() - 1;
                        }
                        flags[0] = true;
                    }
                    if ((scalar3 > val) != (scalar1 > val)) {
                        count++;
                        double proportion = (scalar3 - val) / (scalar3 - scalar1);
                        Vector3d intersection = proportion * p1 + (1 - proportion) * p3;
                        if (fromTwoIDs2IDofNewVert.find(make_pair(min(id3, id1), max(id3, id1))) ==
                            fromTwoIDs2IDofNewVert.end()) {
                            verts_copy.emplace_back(intersection);
                            fromTwoIDs2IDofNewVert[make_pair(min(id3, id1), max(id3, id1))] = verts_copy.size() - 1;
                        }
                        flags[1] = true;
                    }

                    if (count != 1 && count != 2)
                        continue;

                    splitedFaces.insert(index);
                    //  ��ֵ�߹�������
                    if (count == 2) {
                        //splitedFaces.insert(index);

                        if (flags[0] && flags[1]) {
                            // ����p2p3��p1p3
                            auto pre = fromTwoIDs2IDofNewVert[make_pair(min(id2, id3), max(id2, id3))];
                            auto nxt = fromTwoIDs2IDofNewVert[make_pair(min(id3, id1), max(id3, id1))];

                            if (scalar3 < val) {
                                faceLess.emplace_back(Eigen::Vector3i(pre, id3, nxt));
                                faceLarger.emplace_back(Eigen::Vector3i(id1, pre, nxt));
                                faceLarger.emplace_back(Eigen::Vector3i(id1, id2, pre));
                            } else {
                                faceLarger.emplace_back(Eigen::Vector3i(pre, id3, nxt));
                                faceLess.emplace_back(Eigen::Vector3i(id1, pre, nxt));
                                faceLess.emplace_back(Eigen::Vector3i(id1, id2, pre));
                            }
                        } else if (flags[1] && flags[2]) {
                            // ����p2p3��p1p3
                            auto pre = fromTwoIDs2IDofNewVert[make_pair(min(id3, id1), max(id3, id1))];
                            auto nxt = fromTwoIDs2IDofNewVert[make_pair(min(id1, id2), max(id1, id2))];

                            if (scalar1 < val) {
                                faceLess.emplace_back(Eigen::Vector3i(pre, id1, nxt));
                                faceLarger.emplace_back(Eigen::Vector3i(id2, pre, nxt));
                                faceLarger.emplace_back(Eigen::Vector3i(id2, id3, pre));
                            } else {
                                faceLarger.emplace_back(Eigen::Vector3i(pre, id1, nxt));
                                faceLess.emplace_back(Eigen::Vector3i(id2, pre, nxt));
                                faceLess.emplace_back(Eigen::Vector3i(id2, id3, pre));
                            }
                        } else if (flags[2] && flags[0]) {
                            // ����p2p3��p1p3
                            auto pre = fromTwoIDs2IDofNewVert[make_pair(min(id1, id2), max(id1, id2))];
                            auto nxt = fromTwoIDs2IDofNewVert[make_pair(min(id2, id3), max(id2, id3))];

                            if (scalar2 < val) {
                                faceLess.emplace_back(Eigen::Vector3i(pre, id2, nxt));
                                faceLarger.emplace_back(Eigen::Vector3i(id3, pre, nxt));
                                faceLarger.emplace_back(Eigen::Vector3i(id3, id1, pre));
                            } else {
                                faceLarger.emplace_back(Eigen::Vector3i(pre, id2, nxt));
                                faceLess.emplace_back(Eigen::Vector3i(id3, pre, nxt));
                                faceLess.emplace_back(Eigen::Vector3i(id3, id1, pre));
                            }
                        }
                    }
                    // ��ֵ�߹�һ������һ������
                    if (count == 1) {
                        if (flags[0] && scalar1 == val) // ����p1�ͱ�[p2,p3]
                        {
                            auto new_id = fromTwoIDs2IDofNewVert[make_pair(min(id2, id3), max(id2, id3))];
                            if (scalar2 < val) {
                                faceLess.emplace_back(Eigen::Vector3i(id1, id2, new_id));
                                faceLarger.emplace_back(Eigen::Vector3i(id1, new_id, id3));
                            } else {
                                faceLarger.emplace_back(Eigen::Vector3i(id1, id2, new_id));
                                faceLess.emplace_back(Eigen::Vector3i(id1, new_id, id3));
                            }
                        } else if (flags[1] && scalar2 == val) // ����p2�ͱ�[p3,p1]
                        {
                            auto new_id = fromTwoIDs2IDofNewVert[make_pair(min(id3, id1), max(id3, id1))];
                            if (scalar1 < val) {
                                faceLess.emplace_back(Eigen::Vector3i(id1, id2, new_id));
                                faceLarger.emplace_back(Eigen::Vector3i(id3, new_id, id2));
                            } else {
                                faceLarger.emplace_back(Eigen::Vector3i(id1, id2, new_id));
                                faceLess.emplace_back(Eigen::Vector3i(id3, new_id, id2));
                            }
                        } else if (flags[2] && scalar3 == val) // ����p3�ͱ�[p1,p2]
                        {
                            auto new_id = fromTwoIDs2IDofNewVert[make_pair(min(id1, id2), max(id1, id2))];
                            if (scalar1 < val) {
                                faceLess.emplace_back(Eigen::Vector3i(id3, id1, new_id));
                                faceLarger.emplace_back(Eigen::Vector3i(id3, new_id, id2));
                            } else {
                                faceLarger.emplace_back(Eigen::Vector3i(id3, id1, new_id));
                                faceLess.emplace_back(Eigen::Vector3i(id3, new_id, id2));
                            }
                        }
                    }
                }
            }

            for (int index = 0; index < modelFaces.size(); index++) {
                if (splitedFaces.find(index) == splitedFaces.end()) {
                    if (scalarField[modelFaces[index].x()] > val) {
                        faceLarger.emplace_back(modelFaces[index]);
                    } else
                        faceLess.emplace_back(modelFaces[index]);
                }

            }

            // ���ĵ�����
            std::vector<Vector3d> vertsLarger; // >valģ��
            map<int, int> fromOldID2NewIDforLarger;
            for (int index = 0; index < faceLarger.size(); index++) {
                auto t = faceLarger[index];
                int oldIDs[] = {t.x(), t.y(), t.z()};
                for (int i = 0; i < 3; i++) {
                    int id = oldIDs[i];
                    if (fromOldID2NewIDforLarger.find(id) == fromOldID2NewIDforLarger.end()) {
                        vertsLarger.emplace_back(verts_copy[id]);
                        fromOldID2NewIDforLarger[id] = vertsLarger.size() - 1;
                    }
                }
                faceLarger[index] = Eigen::Vector3i(fromOldID2NewIDforLarger[oldIDs[0]],
                                                    fromOldID2NewIDforLarger[oldIDs[1]],
                                                    fromOldID2NewIDforLarger[oldIDs[2]]);
            }

            std::vector<Vector3d> vertsLess; // <valģ��
            map<int, int> fromOldID2NewIDforLess;
            for (int index = 0; index < faceLess.size(); index++) {
                auto t = faceLess[index];
                int oldIDs[] = {t.x(), t.y(), t.z()};
                for (int i = 0; i < 3; i++) {
                    int id = oldIDs[i];
                    if (fromOldID2NewIDforLess.find(id) == fromOldID2NewIDforLess.end()) {
                        vertsLess.emplace_back(verts_copy[id]);
                        fromOldID2NewIDforLess[id] = vertsLess.size() - 1;
                    }
                }
                faceLess[index] = Eigen::Vector3i(fromOldID2NewIDforLess[oldIDs[0]], fromOldID2NewIDforLess[oldIDs[1]],
                                                  fromOldID2NewIDforLess[oldIDs[2]]);
            }

            return make_pair(BaseModel(vertsLarger, faceLarger), BaseModel(vertsLess, faceLess));
        }

        MatrixXd BaseModel::getClosestPoint(const MatrixXd &queryPointMat) const {
            Eigen::VectorXd sqrD;
            Eigen::VectorXi I;
            Eigen::MatrixXd C;

            // the output sqrD contains the (unsigned) squared distance from each point in P
            // to its closest point given in C which lies on the element in F given by I
            aabbTree.squared_distance(m_V, m_F, queryPointMat, sqrD, I, C);

            return C;
        }

        inline MatrixXd BaseModel::getSurfacePointNormal(const MatrixXd &queryPointMat) {
            const size_t numPoint = queryPointMat.rows();

            Eigen::VectorXd sqrD;
            Eigen::VectorXi I;
            Eigen::MatrixXd C;
            getClosestPoint(queryPointMat);

            Eigen::MatrixXd resNormal(numPoint, 3);
            for (int i = 0; i < numPoint; ++i) {
                Eigen::Vector3d normal = m_FN.row(I(i)).normalized();
                resNormal.row(i) = normal;
            }
            return resNormal;
        }

        //////////////////////
        //    I/O: Model    //
        //////////////////////
        void BaseModel::readFile(const std::string &filename) {
            if (!igl::read_triangle_mesh(filename, m_V, m_F)) {
                fprintf(stderr, "[I/O] Error: File %s could not open!", filename.c_str());
                exit(EXIT_FAILURE);
            }
            modelName = str_util::getFileName(filename);
        }

        void BaseModel::setModelAttributeVector() {
            for (int i = 0; i < m_V.rows(); i++) modelVerts.emplace_back(m_V.row(i));
            for (int i = 0; i < m_F.rows(); i++) {
                modelFaces.emplace_back(m_F.row(i));

                modelTris.emplace_back(modelVerts[(m_F.row(i))[0]],
                                       modelVerts[(m_F.row(i))[1]],
                                       modelVerts[(m_F.row(i))[2]]
                );
            }

            nModelVerts = modelVerts.size(), nModelTris = modelFaces.size();

            igl::per_vertex_normals(m_V, m_F, m_VN);
            igl::per_face_normals_stable(m_V, m_F, m_FN);

            aabbTree.init(m_V, m_F);
        }

        void BaseModel::readOffFile(const std::string &filename) {
            std::ifstream in(filename);
            if (!in.is_open()) {
                std::cout << "Unable to open the file!" << std::endl;
            }
            std::string line;
            in >> line;  // off�ַ���
            int VertexNum, FaceNum, EdgeNum;
            in >> VertexNum >> FaceNum >> EdgeNum;   // ��ȡ���㡢�桢�ߵ�����

            for (size_t i = 0; i < VertexNum; i++) {
                double x, y, z;
                in >> x >> y >> z;
                modelVerts.emplace_back(Vector3d(x, y, z));
                //cout << x << " " << y << " " << z << endl;
            }

            for (size_t i = 0; i < FaceNum; i++) {

                int EdgeNumOfFace;   // ������Ƭ��״
                in >> EdgeNumOfFace;

                std::vector<int> indices;
                for (int j = 0; j < EdgeNumOfFace; j++) {

                    int index;
                    in >> index;
                    indices.emplace_back(index);
                }
                for (int j = 1; j < EdgeNumOfFace - 1; j++) {
                    modelFaces.emplace_back(Vector3i(indices[0], indices[j], indices[j + 1]));
                    //cout << indices[0] <<" " << indices[j] << " " << indices[j + 1] << endl;
                }
            }
            in.close();
        }

        void BaseModel::readObjFile(const std::string &filename) {
            std::ifstream in(filename);
            if (!in) {
                fprintf(stderr, "Unable to open the File: %s !\n");
                exit(-1);
            }

            char buf[1024];
            while (in.getline(buf, sizeof(buf)))   // �����ļ�����buf
            {
                std::stringstream line(buf);         // ��line��Ϊ�ַ���
                std::string word;

                line >> word;                  // wordΪ��һ���ַ������Կո�Ϊ��ʶ
                if (word == "v") {
                    double x, y, z;
                    line >> x >> y >> z;
                    modelVerts.emplace_back(Vector3d(x, y, z));

                    //cout << "v " << x << y << z << endl;
                } else if (word == "f") {
                    std::vector<int> indices;
                    int index;

                    while (line >> index) {
                        indices.emplace_back(index);
                    }
                    for (int i = 1; i < indices.size() - 1; i++) {
                        modelFaces.emplace_back(Eigen::Vector3i(indices[0] - 1, indices[i] - 1, indices[i + 1] - 1));
                    }
                }
            }
            in.close();
        }

        void BaseModel::writeObjFile(const std::string &filename) const {
            std::ofstream out(filename);
            out << "# Vertices: " << modelVerts.size() << "\tFaces: " << modelFaces.size() << std::endl;
            for (auto v: modelVerts) {
                out << "v " << v.x() << " " << v.y() << " " << v.z() << " " << std::endl;
            }
            for (auto f: modelFaces) {
                out << "f " << (f + Vector3i(1, 1, 1)).transpose() << std::endl;
            }
            out.close();
        }

        void BaseModel::writeObjFile(const std::string &filename, const std::vector<Vector3d> &V,
                                     const std::vector<Vector3i> &F) const {
            std::ofstream out(filename);
            out << "# Vertices: " << modelVerts.size() << "\tFaces: " << modelFaces.size() << std::endl;
            for (size_t i = 0; i < V.size(); i++) {
                Vector3d v = V[i];
                out << "v " << std::fixed << std::setprecision(5) << v.x() << " " << v.y() << " " << v.z() << " "
                    << std::endl;
            }
            for (size_t i = 0; i < F.size(); i++) {
                Vector3i f = F[i];
                out << "f " << (f + Vector3i(1, 1, 1)).transpose() << std::endl;
            }
            out.close();
        }

        void BaseModel::saveVertices(const std::string &filename, const std::vector<Vector3d> &verts) {
            std::ofstream out(filename);
            out << "# " << verts.size() << " vertices " << std::endl;

            for (const Vector3d &v: verts) {
                out << "v " << std::setiosflags(std::ios::fixed) << std::setprecision(10) << v.x() << " " << v.y()
                    << " "
                    << v.z() << std::endl;
            }
            out.close();
        }

        void
        BaseModel::saveIsoline(const std::string &filename, const std::vector<std::vector<Vector3d>> &isoline) const {
            std::ofstream out(filename);
            out << "g 3d_lines" << std::endl;
            int cnt(0);
            for (auto loop: isoline) {
                /*for (auto v : loop)
                {
                    out << "v " << v.transpose() << endl;
                }*/
                for (int i = 0; i < loop.size(); i++) {
                    out << "v " << loop[i].x() << " " << loop[i].y() << " " << loop[i].z() << std::endl;
                }
                //out << "l ";
                for (int i = 1; i <= loop.size() - 1; i++) {
                    out << "l " << i + cnt << " " << i + 1 + cnt << std::endl;;
                }
                out << "l " << loop.size() << " " << 1 << std::endl;
                //out << 1 + cnt << endl;
                cnt += loop.size();
            }

            out.close();
        }

        void BaseModel::writeTexturedObjFile(const std::string &filename, const std::vector<PDD> &uvs) const {
            std::ofstream out(filename);
            out << "# Vertices: " << modelVerts.size() << "\tFaces: " << modelFaces.size() << std::endl;
            out << "mtllib defaultmaterial.mtl" << std::endl;
            out << "usemtl mydefault" << std::endl;
            for (auto v: modelVerts) {
                out << "v " << v.x() << " " << v.y() << " " << v.z() << " " << std::endl;
            }
            for (auto uv: uvs) {
                out << "vt " << uv.first << " " << uv.second << " " << std::endl;
            }
            for (auto f: modelFaces) {
                auto ids = (f + Eigen::Vector3i(1, 1, 1)).transpose();
                out << "f " << ids.x() << "/" << ids.x() << " " << ids.y() << "/" << ids.y() << " " << ids.z() << "/"
                    << ids.z() << std::endl;
            }
            out.close();
        }

        void BaseModel::writeTexturedObjFile(const std::string &filename, const VectorXd &uvs) const {
            std::ofstream out(filename);
            out << "# Vertices: " << modelVerts.size() << "\tFaces: " << modelFaces.size() << std::endl;
            /*out << "mtllib defaultmaterial.mtl" << std::endl;
            out << "usemtl mydefault" << std::endl;*/
            for (auto v: modelVerts) {
                out << "v " << v.x() << " " << v.y() << " " << v.z() << " " << std::endl;
            }
            // �������������0-1֮��
            //auto maxU = *max_element(uvs.begin(), uvs.end());
            double maxU = uvs.maxCoeff();
            double minU = uvs.minCoeff();
            for (auto u: uvs) {
                out << "vt " << (u - minU) / (maxU - minU) << " " << 0 << " " << std::endl;
            }
            int i = 0;
            for (auto f: modelFaces) {
                Vector3i ids = f + Vector3i(1, 1, 1);
                out << "f " << ids.x() << "/" << ids.x() << " " << ids.y() << "/" << ids.y() << " " << ids.z() << "/"
                    << ids.z() << std::endl;
            }
            out.close();
        }

    }
NAMESPACE_END(ITS)