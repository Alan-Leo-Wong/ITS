#pragma once

#include "Config.hpp"
#include <Eigen/Dense>

NAMESPACE_BEGIN(ITS)
    namespace core {
        using namespace Eigen;

        inline CUDA_GENERAL_CALL double
        bSpineBasis(double x, double nodePos, double nodeWidth) {
            if (fabs(x - (nodePos - nodeWidth)) < 1e-9 || x < nodePos - nodeWidth ||
                fabs(x - (nodePos + nodeWidth)) < 1e-9 || x > nodePos + nodeWidth)
                return 0.0;
            if (x <= nodePos) return 1 + (x - nodePos) / nodeWidth;
            if (x > nodePos) return 1 - (x - nodePos) / nodeWidth;
        }

        inline CUDA_GENERAL_CALL double
        bSplineForPoint(const Vector3d &nodePos, const Vector3d &nodeW, const Vector3d &queryPoint) {
            double x = bSpineBasis(queryPoint.x(), nodePos.x(), nodeW.x());
            if (x <= 0.0) return 0.0;
            double y = bSpineBasis(queryPoint.y(), nodePos.y(), nodeW.y());
            if (y <= 0.0) return 0.0;
            double z = bSpineBasis(queryPoint.z(), nodePos.z(), nodeW.z());
            if (z <= 0.0) return 0.0;
            return x * y * z;
        }

        inline CUDA_GENERAL_CALL double
        bSplineForPoint(const Vector3d &nodePos, double nodeW, const Vector3d &queryPoint) {
            return bSplineForPoint(nodePos, Vector3d(nodeW, nodeW, nodeW), queryPoint);
        }

        inline CUDA_GENERAL_CALL double
        bSplineGradient(double x, double nodePos, double nodeWidth) {
            if (fabs(x - (nodePos - nodeWidth)) < 1e-9 || x < nodePos - nodeWidth ||
                fabs(x - (nodePos + nodeWidth)) < 1e-9 || x > nodePos + nodeWidth)
                return 0.0;
            if (x <= nodePos) return 1 / nodeWidth;
            if (x > nodePos) return -1 / nodeWidth;
        }

        inline CUDA_GENERAL_CALL Vector3d
        bSplineGradientForPoint(const Vector3d &nodePos, const Vector3d &nodeW, const Vector3d &queryPoint) {
            Vector3d gradient;

            gradient.x() = bSplineGradient(queryPoint.x(), nodePos.x(), nodeW.x())
                           * bSpineBasis(queryPoint.y(), nodePos.y(), nodeW.y())
                           * bSpineBasis(queryPoint.z(), nodePos.z(), nodeW.z());

            gradient.y() = bSpineBasis(queryPoint.x(), nodePos.x(), nodeW.x())
                           * bSplineGradient(queryPoint.y(), nodePos.y(), nodeW.y())
                           * bSpineBasis(queryPoint.z(), nodePos.z(), nodeW.z());

            gradient.z() = bSpineBasis(queryPoint.x(), nodePos.x(), nodeW.x())
                           * bSpineBasis(queryPoint.y(), nodePos.y(), nodeW.y())
                           * bSplineGradient(queryPoint.z(), nodePos.z(), nodeW.z());

            return gradient;
        }

        inline CUDA_GENERAL_CALL Vector3d
        bSplineGradientForPoint(const Vector3d &nodePos, double nodeW, const Vector3d &queryPoint) {
            return bSplineGradientForPoint(nodePos, Vector3d(nodeW, nodeW, nodeW), queryPoint);
        }

        inline CUDA_GENERAL_CALL MatrixXd
        de_BaseFunction4PointMat(const MatrixXd &nodePos, const Vector3d &nodeW, const MatrixXd &queryPointMat) {
            MatrixXd gradientMat(queryPointMat.rows(), 3);
            for (int i = 0; i < queryPointMat.rows(); ++i)
                gradientMat.row(i) = bSplineGradientForPoint(nodePos.row(i), nodeW(i), queryPointMat);
            return gradientMat;
        }

    }
NAMESPACE_END(ITS)