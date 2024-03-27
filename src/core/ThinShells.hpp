#pragma once

#include "SVO.hpp"
#include "Mesh.hpp"
#include "utils/File.hpp"
#include "test/TestConfig.h"

NAMESPACE_BEGIN(ITS)
    namespace core {
        using namespace Eigen;
        using namespace svo;
        using namespace utils;
        using std::string;
        using std::vector;

        class ThinShells : public Mesh {
            using test_type = Test::type;
        public:
            ThinShells() = default;

            ThinShells(const string &filename, int _grid, bool lazyTag = false);

            ThinShells(const string &filename, const Vector3i &_grid, bool lazyTag = false);

            ~ThinShells() noexcept = default;

        private:
            [[nodiscard]] Vector3i
            getPointOffset(const Vector3d &modelVert, const Vector3d &origin, const Vector3d &width) const;

            [[nodiscard]] Vector3i
            getPointOffset(const Vector3d &modelVert, const Vector3d &origin, const double &width) const;

            /**
             * Compute the intersections between mesh and svo(fine nodes).
             */
            void cpIntersectionPoints();

            /**
             * Compute the sdf of all svo grid points.
             */
            void cpSDFOfTreeNodes();

            /**
             * Compute B-Spline basis coefficients.
             */
            void cpCoefficients();

            /**
             * Compute the B-Spline value of all extreme points
             * to get the inner and outer shell iso-value
             */
            void cpLatentBSplineValue();

        public:
            /**
             * [API]: Main method for user calling.
             */
            void creatShell();

            /**
             * [API]: Get the iso-value of the inner shell and outer shell.
             * @return
             */
            std::array<double, 2> getShellIsoVal() { return {innerShellIsoVal, outerShellIsoVal}; }

        private:
            /**
             * Save the intersections.
             * @param filename save file
             * @param intersections
             */
            void saveIntersections(const string &filename, const vector<Vector3d> &intersections) const;

            void prepareTestDS();

        public:
            /**
             * [API]: Save the sparse voxel octree depth by depth.
             * @param filename save file
             */
            void saveTree(const string &filename) const;

            /**
             * [API]: Save the intersections.
             * @param filename_1 save file
             * @param filename_2 save file
             */
            void saveIntersections(const string &filename_1, const string &filename_2) const;

            /**
             * [API]: Save the sdf value of all svo grid points.
             * @param filename save file
             */
            void saveSDFValue(const string &filename) const;

            /**
             * [API]: Save the B-Spline coefficients of all svo grid points.
             * @param filename save file
             */
            void saveCoefficients(const string &filename) const;

            /**
             * [API]: Save the extreme point.
             * @param filename save file
             */
            void saveLatentPoint(const string &filename) const;

            /**
             * [API]: Save the B-Spline value of all svo grid points.
             * @param filename save file
             */
            void saveBSplineValue(const string &filename) const;

        public:
            /**
             *
             * [API]: Running marching-cubes algorithm for visualization.
             * @param innerFilename save file of inner shell
             * @param innerResolution mc resolution of inner shell
             * @param outerFilename save file of outer shell
             * @param outerResolution mc resolution of outer shell
             * @param isoFilename save file of zero iso-value surface
             * @return results of marching-cubes
             */
            const std::vector<std::pair<Mesh, string>> &
            mcVisualization(const string &innerFilename, const Vector3i &innerResolution,
                            const string &outerFilename, const Vector3i &outerResolution,
                            const string &isoFilename, const Vector3i &isoResolution);

            /**
             * [API]: Texture mapping of the B-Spline value.
             * @param filename save file
             */
            void textureVisualization(const string &filename) const;

        public:
            /**
             * [API]: Check a point is whether in the range of two shells
             * @param out_file save the result(point in different colors)
             * @param point query point
             */
            void singlePointQuery(const string &out_file, const Vector3d &point);

            /**
             * [API]: Unit test. Check the points are whether in the range of two shells
             * @param points
             * @param time
             * @param session
             * @param choice
             * @return 1: point is outside the outer shell, 2: point is inside the outer shell, 3: otherwise.
             */
            std::vector<int>
            multiPointQuery(const std::vector<Vector3d> &points, double &time, const int &session,
                            const test_type &choice = Test::CPU);

            /**
             * [API]: Check the points are whether in the range of two shells
             * @param out_file save the result(points in different colors)
             * @param points query points
             */
            void multiPointQuery(const string &out_file, const std::vector<Vector3d> &points);

            /**
             * [API]: Check the points are whether in the range of two shells
             * @param out_file save the result(points in different colors)
             * @param points query points
             */
            void multiPointQuery(const string &out_file, const MatrixXd &points);

            /**
             * [API]: Compute the B-Spline value of query points.
             * @param queryPoint
             * @param dummy
             * @return
             */
            double getPointBSplineVal(const Vector3d &queryPoint, bool dummy) const;

            /**
             * [API]: Compute the B-Spline value of query points.
             * @param queryPointMat
             * @return
             */
            VectorXd getPointBSplineVal(const MatrixXd &queryPointMat);

        private:
            Vector3d modelOrigin;

        private:
            /**
             * SVO related data structure.
             */
            SparseVoxelOctree svo;
            Vector3i svo_gridSize;
            double voxelWidth;

            std::vector<Vector3d> nodeWidthArray;
            std::map<uint32_t, uint32_t> morton2FineNodeIdx;

            std::vector<std::map<uint32_t, uint32_t>> depthMorton2Nodes;
            std::vector<std::map<Vector3d, size_t>> depthVert2Idx;

        private:
            std::vector<Vector3d> edgeInterPoints; // Intersection points of octree node and mesh's edges
            std::vector<Vector3d> faceInterPoints; // Intersection points of octree node's edges and mesh's faces
            std::vector<Vector3d> allInterPoints;  // All intersection points of octree node and mesh

        private:
            VectorXd sdfVal; // Real sdf value of all svo grid points
            VectorXd lambda; // Coefficients of B-Spline basis
            VectorXd bSplineVal; // B-Spline value of all svo grid points

        private:
            double innerShellIsoVal = -DINF;
            double outerShellIsoVal = DINF;

        public:
            int treeDepth;
        };

    }
NAMESPACE_END(ITS)