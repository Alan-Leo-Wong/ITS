#include "Mesh.hpp"
#include "utils/IO.hpp"
#include "utils/Common.hpp"
#include "utils/String.hpp"
#include "CUDACompute.hpp"
#include <sstream>
#include <iomanip>
#include <utility>
#include <igl/writeOBJ.h>
#include <igl/read_triangle_mesh.h>
#include <igl/per_vertex_normals.h>
#include <igl/point_mesh_squared_distance.h>

NAMESPACE_BEGIN(ITS)
    namespace core {

        //////////////////////
        //   Constructors   //
        //////////////////////
        Mesh::Mesh(const std::string &filename, bool lazyTag) {
            readMesh(filename);
            if (!lazyTag) updateInternalData();
        }

        Mesh::Mesh(MatrixXd verts, MatrixXi faces) : vertMat(std::move(verts)),
                                                     faceMat(std::move(faces)) {
            updateInternalData();
        }

        //////////////////////
        //   Internal data  //
        //////////////////////
        void Mesh::setModelBaseAttributes() {
            nModelVerts = vertMat.rows();
            nModelTris = faceMat.rows();

            vertVec.clear();
            vertVec.resize(nModelVerts);

#pragma omp parallel for
            for (int i = 0; i < nModelVerts; i++)
                vertVec[i] = vertMat.row(i);

            faceVec.clear();
            trisVec.clear();
            faceVec.resize(nModelTris);
            trisVec.resize(nModelTris);
#pragma omp parallel for
            for (int i = 0; i < nModelTris; i++) {
                faceVec[i] = faceMat.row(i);
                trisVec[i] = Triangle<Vector3d>(vertVec[(faceMat.row(i))[0]],
                                                vertVec[(faceMat.row(i))[1]],
                                                vertVec[(faceMat.row(i))[2]]);
            }

            igl::per_vertex_normals(vertMat, faceMat, vertNormalMat);
            igl::per_face_normals_stable(vertMat, faceMat, faceNormalMat);

            aabbTree.init(vertMat, faceMat);
        }

        void Mesh::setTriAttributes() {
            cuAcc::launch_modelTriAttributeKernel(nModelTris, trisVec);
        }

        void Mesh::setBoundingBox() {
            Vector3d minV = vertMat.colwise().minCoeff();
            Vector3d maxV = vertMat.colwise().maxCoeff();

            modelBoundingBox = AABox<Vector3d>(minV, maxV);
        }

        void Mesh::setUniformBoundingBox() {
            Vector3d minV = vertMat.colwise().minCoeff();
            Vector3d maxV = vertMat.colwise().maxCoeff();

            modelBoundingBox = AABox<Vector3d>(minV, maxV); // initialize answer
            Vector3d lengths = maxV - minV; // check length of given bbox in every direction
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
            Vector3d epsilon = (modelBoundingBox.boxEnd - modelBoundingBox.boxOrigin) / 11; // ????10001
            modelBoundingBox.boxOrigin -= epsilon;
            modelBoundingBox.boxEnd += epsilon;
            modelBoundingBox.boxWidth = modelBoundingBox.boxEnd - modelBoundingBox.boxOrigin;
        }

        void Mesh::updateInternalData() {
            setModelBaseAttributes();
            setTriAttributes();
            setUniformBoundingBox();
        }

        //////////////////////
        //     Utilities    //
        //////////////////////
        void Mesh::addNoise(double noisePercentage, double min_val, double max_val) {
            noiseDir = (std::string) "noise_" + std::to_string(noisePercentage * 100.0);

            // set the number of vertices to be disturbed
            uint numNoisyVerts = static_cast<int>(noisePercentage * nModelVerts);

            std::random_device rd;
            std::mt19937 gen(rd());
            // set the disturbance range
            std::uniform_real_distribution<double> dis(min_val, max_val);

            for (int i = 0; i < numNoisyVerts; i++) {
                int nodeIndex = std::rand() % nModelVerts;

                vertMat.row(nodeIndex) += dis(gen) * RowVector3d::Random();
            }

            updateInternalData();
        }

        std::vector<Vector2i> Mesh::extractEdges() {
            std::cout << "Extracting edges from " << std::quoted(modelName) << std::endl;

            std::vector<Vector2i> edges;
            std::set<PII> uset;

            for (Vector3i f: faceVec) {
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

        //////////////////////
        //   Transformers   //
        //////////////////////
        Matrix3d Mesh::calcScaleMatrix(double scaleFactor) {
            RowVector3d boxMin = vertMat.colwise().minCoeff();
            RowVector3d boxMax = vertMat.colwise().maxCoeff();

            // Get the target solveRes (along the largest dimension)
            double scale = boxMax[0] - boxMin[0];
            scale *= scaleFactor;
            Eigen::Matrix3d zoomMatrix;
            for (int i = 0; i < 3; i++)
                zoomMatrix(i, i) = scale;
            return zoomMatrix;
        }

        Matrix4d Mesh::calcUnitCubeTransformMatrix(double scaleFactor) {
            RowVector3d boxMin = vertMat.colwise().minCoeff();
            RowVector3d boxMax = vertMat.colwise().maxCoeff();

            // Get the target solveRes (along the largest dimension)
            double scale = boxMax[0] - boxMin[0];
            double minScale = scale;
            for (int d = 1; d < 3; d++) {
                scale = std::max<double>(scale, boxMax[d] - boxMin[d]);
                minScale = std::min<double>(scale, boxMax[d] - boxMin[d]);
            }
            // std::cout << 1.1 + scale / minScale << std::endl;
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

        Matrix4d Mesh::calcTransformMatrix(double scaleFactor) {
            RowVector3d boxMin = vertMat.colwise().minCoeff();
            RowVector3d boxMax = vertMat.colwise().maxCoeff();

            // Get the target solveRes (along the largest dimension)
            double scale = boxMax[0] - boxMin[0];
            for (int d = 1; d < 3; d++)
                scale = std::max<double>(scale, boxMax[d] - boxMin[d]);
            scale *= scaleFactor;
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

        void Mesh::model2UnitCube(double scaleFactor) {
            uniformDir = "uniform";
            unitScaleFactor = scaleFactor;

            auto transMat = calcUnitCubeTransformMatrix(scaleFactor);
#pragma omp parallel for
            for (int i = 0; i < vertMat.rows(); ++i) {
                vertMat.row(i) += transMat.block(3, 0, 1, 3);
                vertMat.row(i) = vertMat.row(i) * transMat.block(0, 0, 3, 3);
            }

            updateInternalData();
        }

        void Mesh::unitCube2Model() {
            uniformDir = "non-uniform";

            auto transMat = calcUnitCubeTransformMatrix(unitScaleFactor);
            Matrix3d inverseTrans = transMat.block(0, 0, 3, 3).inverse();
#pragma omp parallel for
            for (int i = 0; i < vertMat.rows(); ++i) {
                vertMat.row(i) = vertMat.row(i) * inverseTrans;
                vertMat.row(i) -= transMat.block(3, 0, 1, 3);
            }

            updateInternalData();
        }

        void Mesh::zoomModel(double scaleFactor) {
            Matrix3d zoomMat = calcScaleMatrix(scaleFactor);
            vertMat = vertMat * zoomMat;

            updateInternalData();
        }

        void Mesh::transformModel(double scaleFactor) {
            Matrix4d transMat = calcTransformMatrix(scaleFactor);
#pragma omp parallel for
            for (int i = 0; i < vertMat.rows(); ++i) {
                vertMat.row(i) += transMat.block(3, 0, 1, 3);
                vertMat.row(i) = vertMat.row(i) * transMat.block(0, 0, 3, 3);
            }
            updateInternalData();
        }

        Eigen::MatrixXd
        Mesh::generateGaussianRandomPoints(const size_t &numPoints, const float &_scaleFactor, const float &dis) {
            Eigen::MatrixXd M;
            const Eigen::RowVector3d min_area = modelBoundingBox.boxOrigin;
            const Eigen::RowVector3d max_area = modelBoundingBox.boxEnd;
            donut::getGaussianRandomMatrix<double>(min_area, max_area, numPoints, _scaleFactor, dis, M);
            return M;
        }

        Eigen::MatrixXd
        Mesh::generateUniformRandomPoints(const size_t &numPoints, const float &_scaleFactor, const float &dis) {
            Eigen::MatrixXd M;
            const Eigen::RowVector3d min_area = modelBoundingBox.boxOrigin;
            const Eigen::RowVector3d max_area = modelBoundingBox.boxEnd;
            donut::getUniformRandomMatrix<double>(min_area, max_area, numPoints, _scaleFactor, dis, M);
            return M;
        }

        Eigen::MatrixXd Mesh::generateGaussianRandomPoints(const std::string &filename, const size_t &numPoints,
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

        Eigen::MatrixXd Mesh::generateUniformRandomPoints(const std::string &filename, const size_t &numPoints,
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
        Mesh::generateUniformRandomPoints(const std::string &filename, const size_t &numPoints,
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
        Mesh::generateGaussianRandomPoints(const std::string &filename, const size_t &numPoints,
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

        MatrixXd Mesh::getClosestPoint(const MatrixXd &queryPointMat) const {
            Eigen::VectorXd sqrD;
            Eigen::VectorXi I;
            Eigen::MatrixXd C;

            // the output sqrD contains the (unsigned) squared distance from each point in P
            // to its closest point given in C which lies on the element in F given by I
            aabbTree.squared_distance(vertMat, faceMat, queryPointMat, sqrD, I, C);

            return C;
        }

        MatrixXd Mesh::getSurfacePointNormal(const MatrixXd &queryPointMat) {
            const size_t numPoint = queryPointMat.rows();

            Eigen::VectorXd sqrD;
            Eigen::VectorXi I;
            Eigen::MatrixXd C;
            getClosestPoint(queryPointMat);

            Eigen::MatrixXd resNormal(numPoint, 3);
            for (int i = 0; i < numPoint; ++i) {
                Eigen::Vector3d normal = faceNormalMat.row(I(i)).normalized();
                resNormal.row(i) = normal;
            }
            return resNormal;
        }


        //////////////////////
        //       I / O      //
        //////////////////////
        void Mesh::readMesh(const std::string &filename) {
            if (!igl::read_triangle_mesh(filename, vertMat, faceMat)) {
                fprintf(stderr, "[I/O] Error: File %s could not open!", filename.c_str());
                exit(EXIT_FAILURE);
            }
            modelName = str_util::getFileName(filename);
        }

        void Mesh::writeMesh(const std::string &filename) const {
            if (str_util::getFileExtension(filename) != ".obj") {
                std::cerr << "Unsupported file format!\n";
                return;
            }
            igl::writeOBJ(filename, vertMat, faceMat);
        }

        void Mesh::writeTexturedObjFile(const std::string &filename, const std::vector<PDD> &uvs) const {
            std::ofstream out(filename);
            out << "# Vertices: " << vertVec.size() << "\tFaces: " << faceVec.size() << std::endl;
            out << "mtllib defaultmaterial.mtl" << std::endl;
            out << "usemtl mydefault" << std::endl;
            for (auto v: vertVec) {
                out << "v " << v.x() << " " << v.y() << " " << v.z() << " " << std::endl;
            }
            for (auto uv: uvs) {
                out << "vt " << uv.first << " " << uv.second << " " << std::endl;
            }
            for (auto f: faceVec) {
                auto ids = (f + Eigen::Vector3i(1, 1, 1)).transpose();
                out << "f " << ids.x() << "/" << ids.x() << " " << ids.y() << "/" << ids.y() << " " << ids.z() << "/"
                    << ids.z() << std::endl;
            }
            out.close();
        }

        void Mesh::writeTexturedObjFile(const std::string &filename, const VectorXd &uvs) const {
            std::ofstream out(filename);
            out << "# Vertices: " << vertVec.size() << "\tFaces: " << faceVec.size() << std::endl;
            /*out << "mtllib defaultmaterial.mtl" << std::endl;
            out << "usemtl mydefault" << std::endl;*/
            for (auto v: vertVec) {
                out << "v " << v.x() << " " << v.y() << " " << v.z() << " " << std::endl;
            }

            // Texture coordinates need to be between 0 and 1.
            double maxU = uvs.maxCoeff();
            double minU = uvs.minCoeff();
            for (auto u: uvs) {
                out << "vt " << (u - minU) / (maxU - minU) << " " << 0 << " " << std::endl;
            }
            for (const auto &f: faceVec) {
                Vector3i ids = f + Vector3i(1, 1, 1);
                out << "f " << ids.x() << "/" << ids.x() << " " << ids.y() << "/" << ids.y() << " " << ids.z() << "/"
                    << ids.z() << std::endl;
            }
            out.close();
        }

    }
NAMESPACE_END(ITS)