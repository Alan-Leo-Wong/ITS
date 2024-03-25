#pragma once

#include "Config.hpp"
#include <Eigen/Dense>

NAMESPACE_BEGIN(ITS)
    namespace core {
        using namespace Eigen;

        inline CUDA_GENERAL_CALL double
        BaseFunction(double x, double nodePos, double nodeWidth) {
            if (fabs(x - (nodePos - nodeWidth)) < 1e-9 || x < nodePos - nodeWidth ||
                fabs(x - (nodePos + nodeWidth)) < 1e-9 || x > nodePos + nodeWidth)
                return 0.0;
            if (x <= nodePos) return 1 + (x - nodePos) / nodeWidth;
            if (x > nodePos) return 1 - (x - nodePos) / nodeWidth;
        }

        inline CUDA_GENERAL_CALL double
        BaseFunction4Point(const Vector3d &nodePos, const Vector3d &nodeW, const Vector3d &queryPoint) {
            double x = BaseFunction(queryPoint.x(), nodePos.x(), nodeW.x());
            if (x <= 0.0) return 0.0;
            double y = BaseFunction(queryPoint.y(), nodePos.y(), nodeW.y());
            if (y <= 0.0) return 0.0;
            double z = BaseFunction(queryPoint.z(), nodePos.z(), nodeW.z());
            if (z <= 0.0) return 0.0;
            return x * y * z;
        }

        inline CUDA_GENERAL_CALL double
        BaseFunction4Point(const Vector3d &nodePos, double nodeW, const Vector3d &queryPoint) {
            return BaseFunction4Point(nodePos, Vector3d(nodeW, nodeW, nodeW), queryPoint);
        }

        inline CUDA_GENERAL_CALL double
        de_BaseFunction(double x, double nodePos, double nodeWidth) {
            if (fabs(x - (nodePos - nodeWidth)) < 1e-9 || x < nodePos - nodeWidth ||
                fabs(x - (nodePos + nodeWidth)) < 1e-9 || x > nodePos + nodeWidth)
                return 0.0;
            if (x <= nodePos) return 1 / nodeWidth;
            if (x > nodePos) return -1 / nodeWidth;
        }

        inline CUDA_GENERAL_CALL Vector3d
        de_BaseFunction4Point(const Vector3d &nodePos, const Vector3d &nodeW, const Vector3d &queryPoint) {
            Vector3d gradient;

            gradient.x() = de_BaseFunction(queryPoint.x(), nodePos.x(), nodeW.x())
                           * BaseFunction(queryPoint.y(), nodePos.y(), nodeW.y())
                           * BaseFunction(queryPoint.z(), nodePos.z(), nodeW.z());

            gradient.y() = BaseFunction(queryPoint.x(), nodePos.x(), nodeW.x())
                           * de_BaseFunction(queryPoint.y(), nodePos.y(), nodeW.y())
                           * BaseFunction(queryPoint.z(), nodePos.z(), nodeW.z());

            gradient.z() = BaseFunction(queryPoint.x(), nodePos.x(), nodeW.x())
                           * BaseFunction(queryPoint.y(), nodePos.y(), nodeW.y())
                           * de_BaseFunction(queryPoint.z(), nodePos.z(), nodeW.z());

            return gradient;
        }

        inline CUDA_GENERAL_CALL Vector3d
        de_BaseFunction4Point(const Vector3d &nodePos, double nodeW, const Vector3d &queryPoint) {
            return de_BaseFunction4Point(nodePos, Vector3d(nodeW, nodeW, nodeW), queryPoint);
        }

        inline CUDA_GENERAL_CALL MatrixXd
        de_BaseFunction4PointMat(const MatrixXd &nodePos, const Vector3d &nodeW, const MatrixXd &queryPointMat) {
            MatrixXd gradientMat(queryPointMat.rows(), 3);
            for (int i = 0; i < queryPointMat.rows(); ++i)
                gradientMat.row(i) = de_BaseFunction4Point(nodePos.row(i), nodeW(i), queryPointMat);
            return gradientMat;
        }

    }
NAMESPACE_END(ITS)