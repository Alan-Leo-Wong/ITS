﻿#pragma once

#include "SVO.hpp"
#include "Mesh.hpp"
#include "utils/String.hpp"
#include "test/TestConfig.h"

NAMESPACE_BEGIN(ITS)
    namespace core {
        using namespace Eigen;
        using namespace svo;
        using namespace str_util;
        using std::string;
        using std::vector;

        class ThinShells : public Mesh {
            using test_type = Test::type;
        public:
            ThinShells() = default;

            ThinShells(const string& filename, int _grid);

            ThinShells(const string& filename, const Vector3i& _grid);

            ~ThinShells() noexcept = default;

        private:
            [[nodiscard]] Vector3i
            getPointOffset(const Vector3d &modelVert, const Vector3d &origin, const Vector3d &width) const;

            [[nodiscard]] Vector3i
            getPointOffset(const Vector3d &modelVert, const Vector3d &origin, const double &width) const;

            void cpIntersectionPoints();

            void cpSDFOfTreeNodes();

            void cpCoefficients();

            void cpLatentBSplineValue();

        public:
            void creatShell();

            // Octree& bSplineTree() { return bSplineTree; }
            // const Octree& bSplineTree() const { return bSplineTree; }

            std::array<double, 2> getShellIsoVal() { return {innerShellIsoVal, outerShellIsoVal}; }

        public:
            void saveTree(const string &filename) const;

            void saveIntersections(const string &filename, const vector<Vector3d> &intersections) const;

            void saveIntersections(const string &filename_1, const string &filename_2) const;

            void saveSDFValue(const string &filename) const;

            void saveCoefficients(const string &filename) const;

            void saveLatentPoint(const string &filename) const;

            void saveBSplineValue(const string &filename) const;

        public:
            void mcVisualization(const string &innerFilename, const Vector3i &innerResolution,
                                 const string &outerFilename, const Vector3i &outerResolution,
                                 const string &isoFilename, const Vector3i &isoResolution);

            void textureVisualization(const string &filename) const;

        private:
            std::vector<std::map<uint32_t, uint32_t>> depthMorton2Nodes;
            std::vector<std::map<Vector3d, size_t>> depthVert2Idx;

            void prepareTestDS();

            MatrixXd getPointNormal(const MatrixXd &queryPointMat);

            double getPointBSplineVal(const Vector3d &queryPoint, bool dummy) const;

            VectorXd getPointBSplineVal(const MatrixXd &queryPointMat);

        public:
            void singlePointQuery(const string &out_file, const Vector3d &point);

            std::vector<int>
            multiPointQuery(const std::vector<Vector3d> &points, double &time, const int &session,
                            const test_type &choice = Test::CPU);

            void multiPointQuery(const string &out_file, const std::vector<Vector3d> &points);

            void multiPointQuery(const string &out_file, const MatrixXd &points);

        private:
            Vector3d modelOrigin;

            Vector3i svo_gridSize;
            SparseVoxelOctree svo;
            double voxelWidth;
            std::vector<Vector3d> nodeWidthArray;
            //vector<Vector3d> nodeWidthArray;
            std::map<uint32_t, uint32_t> morton2FineNodeIdx;

            std::vector<Vector3d> edgeInterPoints; // Intersection points of octree node and mesh's edges
            std::vector<Vector3d> faceInterPoints; // Intersection points of octree node's edges and mesh's faces
            std::vector<Vector3d> allInterPoints;  // All intersection points of octree node and mesh

        private:
            VectorXd sdfVal;
            VectorXd lambda;
            VectorXd bSplineVal;

        private:
            double innerShellIsoVal = -DINF;
            double outerShellIsoVal = DINF;

        public:
            int treeDepth;
        };

    }
NAMESPACE_END(ITS)