#pragma once

#include "Config.hpp"
#include "ModelDefine.h"
#include "detail/BasicDataType.hpp"
#include "detail/Geometry.hpp"
#include <utility>
#include <igl/AABB.h>

NAMESPACE_BEGIN(ITS)
    namespace core {
        using namespace Eigen;

        /**
         * 3D Triangle Mesh
         */
        class Mesh {
        public:
            std::string uniformDir = "non-uniform";
            std::string noiseDir = "non-noise";

        public:
            Mesh() noexcept = default;

            explicit Mesh(const std::string &filename, bool lazyTag = false);

            Mesh(MatrixXd verts, MatrixXi faces);

            ~Mesh() noexcept = default;

        public:
            [[nodiscard]] std::vector<Vector3d> getModelVerts() const { return vertVec; }

            [[nodiscard]] std::vector<Vector3i> getModelFaces() const { return faceVec; }

            [[nodiscard]] MatrixXd getVertices() const { return vertMat; }

            [[nodiscard]] MatrixXi getFaces() const { return faceMat; }

            [[nodiscard]] std::vector<Triangle<Vector3d>> getModelTris() const { return trisVec; }

        private:
            /**
             * Update internal data like vectors, triangle normals...
             */
            void updateInternalData();

            /**
             * Init data stored in vector.
             */
            void setModelBaseAttributes();

            /**
             * Init the normal and area of triangles.
             */
            void setTriAttributes();

            /**
             * Init axis aligned bounding-box of the model.
             */
            void setBoundingBox();

            /**
             * Init uniform axis aligned bounding-box of the model.
             */
            void setUniformBoundingBox();

        public:
            /**
             * Extract edges from the model.
             * @return All edge pairs, represented by two vertex index
             */
            std::vector<Vector2i> extractEdges();

            /**
             * [API]: Add noise to the model by randomly perturbing vertex positions.
             * @param noisePercentage Perturbation percentage
             * @param min_val Minimum perturbation value
             * @param max_val Maximum perturbation value
             */
            void addNoise(double noisePercentage, double min_val = -0.1, double max_val = 0.1);

        private:
            /**
             * A matrix calculator for scaling the model.
             * @param scaleFactor
             * @return scale matrix
             */
            Matrix3d calcScaleMatrix(double scaleFactor);

            /**
             * A matrix calculator for transforming the model to the unit range: [0, 1]^3.
             * @return transform matrix
             */
            Matrix4d calcUnitCubeTransformMatrix(double scaleFactor);

            /**
             * A general transform matrix calculator, the model is scaled first,
             * and then its new center is moved to the position of
             * the model before scaling.
             * @param scaleFactor
             * @return transform matrix
             */
            Matrix4d calcTransformMatrix(double scaleFactor);

        public:
            /**
             * [API]: Transform the model to the unit range: scaleFactor * [0, 1]^3
             * it will call 'calcUnitCubeTransformMatrix'.
             * @param scaleFactor
             */
            void model2UnitCube(double scaleFactor = 1.0);

            /**
             * [API]: Recover the model position from the 'scaleFactor * [0, 1]^3' range to its world coordinates.
             */
            void unitCube2Model();

            /**
             * [API]: Scaling the model
             */
            void zoomModel(double scaleFactor);

            /**
             * [API]: Transform the model by calling 'calcTransformMatrix'.
             */
            void transformModel(double scaleFactor);

        public:
            Eigen::MatrixXd
            generateGaussianRandomPoints(const size_t &numPoints, const float &_scaleFactor, const float &dis);

            Eigen::MatrixXd
            generateUniformRandomPoints(const size_t &numPoints, const float &_scaleFactor, const float &dis);

            Eigen::MatrixXd
            generateGaussianRandomPoints(const std::string &filename, const size_t &numPoints,
                                         const float &_scaleFactor,
                                         const float &dis);

            Eigen::MatrixXd
            generateUniformRandomPoints(const std::string &filename, const size_t &numPoints, const float &_scaleFactor,
                                        const float &dis);

            std::vector<Vector3d>
            generateUniformRandomPoints(const std::string &filename, const size_t &numPoints,
                                        const double &_scaleFactor,
                                        const Vector3d &dis);

            std::vector<Vector3d>
            generateGaussianRandomPoints(const std::string &filename, const size_t &numPoints,
                                         const double &_scaleFactor,
                                         const Vector3d &dis);

        protected:
            [[nodiscard]] MatrixXd getClosestPoint(const MatrixXd &queryPointMat) const;

            virtual MatrixXd getSurfacePointNormal(const MatrixXd &queryPointMat);

        public:
            void readMesh(const std::string &filename);

            void writeMesh(const std::string &filename) const;

            void writeTexturedObjFile(const std::string &filename, const std::vector<PDD> &uvs) const;

            void writeTexturedObjFile(const std::string &filename, const VXd &uvs) const;

        protected:
            uint nModelVerts = 0;
            uint nModelTris = 0;

        protected:
            MatrixXd vertMat;  // vertex matrix
            MatrixXd vertNormalMat; // vertex normal matrix
            MatrixXi faceMat;  // face(vertex index) matrix
            MatrixXd faceNormalMat; // face normal matrix

            std::vector<Vector3d> vertVec;
            std::vector<Vector3i> faceVec;
            std::vector<Triangle<Vector3d>> trisVec;  // face(vertex coordinates) vector

        protected:
            std::string modelName;
            double unitScaleFactor; // record scale factor when calling 'model2UnitCube'

            AABox<Eigen::Vector3d> modelBoundingBox;
            igl::AABB<MatrixXd, 3> aabbTree;
        };

    }
NAMESPACE_END(ITS)