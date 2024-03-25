#pragma once

#include "Config.hpp"
#include "ModelDefine.h"
#include "detail/BasicDataType.hpp"
#include "detail/Geometry.hpp"
#include <igl/AABB.h>

#include <utility>

NAMESPACE_BEGIN(ITS)
    namespace core {
        using namespace Eigen;

        class BaseModel {
        public:
            std::string uniformDir = "non-uniform";
            std::string noiseDir = "non-noise";

        protected:
            MatrixXd m_V;
            MatrixXd m_VN;
            MatrixXi m_F;
            MatrixXd m_FN;

            bool is2UnitCube;
            double scaleFactor;

            std::vector<Vector3d> modelVerts;
            std::vector<Vector3i> modelFaces;
            std::vector<Triangle<Vector3d>> modelTris;

            igl::AABB<MatrixXd, 3> aabbTree;

        protected:
            std::string modelName;
            AABox<Eigen::Vector3d> modelBoundingBox;

            uint nModelVerts = 0;
            uint nModelTris = 0;

        private:
            void setModelAttributeVector();

        public:
            BaseModel() = default;

            BaseModel(std::vector<Vector3d> verts, std::vector<Vector3i> faces);

            explicit BaseModel(const std::string &filename);

            BaseModel(const std::string &filename, bool _is2UnitCube, double _scaleFactor);

            BaseModel(const std::string &filename, bool _is2UnitCube, double _scaleFactor,
                      bool _isAddNoise, double noisePercentage);

            ~BaseModel() = default;

        public:
            [[nodiscard]] std::vector<Vector3d> getModelVerts() const { return modelVerts; }

            [[nodiscard]] std::vector<Vector3i> getModelFaces() const { return modelFaces; }

            [[nodiscard]] MatrixXd getVertices() const {return m_V;}

            [[nodiscard]] MatrixXi getFaces() const {  return m_F; }

            [[nodiscard]] std::vector<Triangle<Vector3d>> getModelTris() const { return modelTris; }

        private:
            Eigen::Matrix4d calcUnitCubeTransformMatrix();

            Eigen::Matrix4d calcTransformMatrix(const float &_scaleFactor);

            Eigen::Matrix3d calcScaleMatrix();

            void addNoise(const double &noisePercentage, const double &min_val = -0.1, const double &max_val = 0.1);

        public:
            void model2UnitCube();

            void unitCube2Model();

            void zoomModel();

            void transformModel(const float &_scaleFactor);

        public:
            std::vector<Vector2i> extractEdges();

            //void scaleMatrix(MatrixXd V);

            void setBoundingBox(const float &_scaleFactor = 1);

            void setUniformBoundingBox();

            void setTriAttributes();

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

            // ��ȡ��ֵ��
            std::vector<std::vector<Vector3d>>
            extractIsoline(const std::vector<double> &scalarField, const double &val) const;

            // �з�ģ��
            std::pair<BaseModel, BaseModel>
            splitModelByIsoline(const std::vector<double> &scalarField, const double &val) const;

        protected:
            [[nodiscard]] MatrixXd getClosestPoint(const MatrixXd &queryPointMat) const;

            virtual MatrixXd getSurfacePointNormal(const MatrixXd &queryPointMat);

        public:
            void readFile(const std::string &filename);

            void readOffFile(const std::string &filename);

            void readObjFile(const std::string &filename);

            void writeObjFile(const std::string &filename) const;

            void writeObjFile(const std::string &filename, const std::vector<Vector3d> &V,
                              const std::vector<Vector3i> &F) const;

            void writeTexturedObjFile(const std::string &filename, const std::vector<PDD> &uvs) const;

            void writeTexturedObjFile(const std::string &filename, const VXd &uvs) const;

            void saveVertices(const std::string &filename, const std::vector<Vector3d> &verts);

            // �����ֵ��
            void saveIsoline(const std::string &filename, const std::vector<std::vector<Vector3d>> &isoline) const;
        };

    }
NAMESPACE_END(ITS)