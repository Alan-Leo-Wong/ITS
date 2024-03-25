#pragma once

#include "SVO.hpp"
#include "BaseModel.hpp"
//#include "ParticleMesh.h"
#include "utils/String.hpp"
#include "test/TestConfig.h"

NAMESPACE_BEGIN(ITS)
namespace core {
    using namespace Eigen;
    using std::string;

    class ThinShells : public BaseModel/*, public ParticleMesh*/ {
        using test_type = Test::type;
    private:
        Vector3d modelOrigin;

        Vector3i svo_gridSize;
        SparseVoxelOctree svo;
        double voxelWidth;
        std::vector<Vector3d> nodeWidthArray;
        //vector<Vector3d> nodeWidthArray;
        std::map<uint32_t, uint32_t> morton2FineNodeIdx;

        std::vector <Vector3d> edgeInterPoints; // Intersection points of octree node and mesh's edges
        std::vector <Vector3d> faceInterPoints; // Intersection points of octree node's edges and mesh's faces
        std::vector <Vector3d> allInterPoints;  // All intersection points of octree node and mesh

    private:
        VectorXd sdfVal;
        VectorXd lambda;
        VectorXd bSplineVal;

    private:
        double innerShellIsoVal = -DINF;
        double outerShellIsoVal = -DINF;

    public:
        int treeDepth;

    public:
        // constructor and destructor
        ThinShells() {}

        ThinShells(const string &filename, const int &_grid_x, const int &_grid_y, const int &_grid_z) :
                svo_gridSize(_grid_x, _grid_y, _grid_z), BaseModel(filename), modelOrigin(modelBoundingBox.boxOrigin),
                svo(_grid_x, _grid_y, _grid_z) {
            svo.createOctree(nModelTris, modelTris, modelBoundingBox, concatFilePath((string) VIS_DIR, modelName));
            treeDepth = svo.treeDepth;
            voxelWidth = svo.svoNodeArray[0].width;
#ifdef IO_SAVE
            saveTree("");
#endif // !IO_SAVE
        }

        ThinShells(const string &filename, const Vector3i &_grid) : svo_gridSize(_grid), BaseModel(filename),
                                                               modelOrigin(modelBoundingBox.boxOrigin), svo(_grid) {
            svo.createOctree(nModelTris, modelTris, modelBoundingBox, concatFilePath((string) VIS_DIR, modelName));
            treeDepth = svo.treeDepth;
            voxelWidth = svo.svoNodeArray[0].width;
#ifdef IO_SAVE
            saveTree("");
#endif // !IO_SAVE
        }

        ThinShells(const string &filename, const int &_grid_x, const int &_grid_y, const int &_grid_z,
                   const bool &_is2UnitCube, const double &_scaleFactor)
                : svo_gridSize(_grid_x, _grid_y, _grid_z), BaseModel(filename, _is2UnitCube, _scaleFactor),
                  modelOrigin(modelBoundingBox.boxOrigin), svo(_grid_x, _grid_y, _grid_z) {
            svo.createOctree(nModelTris, modelTris, modelBoundingBox, concatFilePath((string) VIS_DIR, modelName));
            treeDepth = svo.treeDepth;
            voxelWidth = svo.svoNodeArray[0].width;
#ifdef IO_SAVE
            saveTree("");
#endif // !IO_SAVE
        }

        ThinShells(const string &filename, const int &_grid_x, const int &_grid_y, const int &_grid_z,
                   const bool &_is2UnitCube, const double &_scaleFactor, const bool &_isAddNoise,
                   const double &noisePercentage)
                : svo_gridSize(_grid_x, _grid_y, _grid_z),
                  BaseModel(filename, _is2UnitCube, _scaleFactor, _isAddNoise, noisePercentage),
                  modelOrigin(modelBoundingBox.boxOrigin), svo(_grid_x, _grid_y, _grid_z) {
            svo.createOctree(nModelTris, modelTris, modelBoundingBox, concatFilePath((string) VIS_DIR, modelName));
            treeDepth = svo.treeDepth;
            voxelWidth = svo.svoNodeArray[0].width;
#ifdef IO_SAVE
            saveTree("");
#endif // !IO_SAVE
        }

        ThinShells(const string &filename, const Vector3i &_grid, const bool &_is2UnitCube, const double &_scaleFactor)
                : svo_gridSize(_grid), BaseModel(filename, _is2UnitCube, _scaleFactor),
                  modelOrigin(modelBoundingBox.boxOrigin), svo(_grid) {
            svo.createOctree(nModelTris, modelTris, modelBoundingBox, concatFilePath((string) VIS_DIR, modelName));
            treeDepth = svo.treeDepth;
            voxelWidth = svo.svoNodeArray[0].width;
#ifndef IO_SAVE
            saveTree("");
#endif // !IO_SAVE
        }

        ~ThinShells() {}

        // ThinShells& operator=(const ThinShells& model);

    private:
        Vector3i getPointDis(const Vector3d &modelVert, const Vector3d &origin, const Vector3d &width) const;

        Vector3i getPointDis(const Vector3d &modelVert, const Vector3d &origin, const double &width) const;

        //void cpIntersectionPoints();
        void cpIntersectionPoints();

        void cpSDFOfTreeNodes();

        void cpCoefficients();

        void cpLatentBSplineValue();

        void initBSplineTree();

        void setLatentMatrix(const double &alpha);

    public:
        void creatShell();

        // Octree& bSplineTree() { return bSplineTree; }
        // const Octree& bSplineTree() const { return bSplineTree; }

        std::array<double, 2> getShellIsoVal() { return {innerShellIsoVal, outerShellIsoVal}; }

    public:
        void saveTree(const string &filename) const;

        void saveIntersections(const string &filename, const vector <Vector3d> &intersections) const;

        void saveIntersections(const string &filename_1, const string &filename_2) const;

        void saveSDFValue(const string &filename) const;

        void saveCoefficients(const string &filename) const;

        void saveLatentPoint(const string &filename) const;

        void saveBSplineValue(const string &filename) const;

    public:
        void mcVisualization(const string &innerFilename, const Vector3i &innerResolution,
                             const string &outerFilename, const Vector3i &outerResolution,
                             const string &isoFilename, const Vector3i &isoResolution) const;

        void textureVisualization(const string &filename) const;

        //friend class CollisionDetection;
    private:
        std::vector<std::map<uint32_t, uint32_t>> depthMorton2Nodes;
        std::vector<std::map<Vector3d, size_t>> depthVert2Idx;

        void prepareTestDS();

        void prepareMoveOnSurface(int &ac_treeDepth,
                                  std::vector<std::vector<Vector3d>> &nodeOrigin,
                                  std::vector<std::map<uint32_t, size_t>> &morton2Nodes,
                                  std::vector<std::vector<std::array<double, 8>>> &nodeBSplineVal,
        std::vector<double> &nodeWidth
        );

        MatrixXd getPointNormal(const MatrixXd &queryPointMat);

        MatrixXd getSurfacePointNormal(const MatrixXd &queryPointMat) override;

        VectorXd getPointBSplineVal(const MatrixXd &queryPointMat);

        std::pair <VectorXd, MatrixXd> getPointValGradient(const MatrixXd &before_queryPointMat, const MatrixXd &queryPointMat);

        MatrixXd getProjectPoint(const MatrixXd &before_queryPointMat, const MatrixXd &queryPointMat, const int &iter);

        void lbfgs_optimization(const int &maxIterations, const std::string &out_file) override;

    public:
        // ���ڱ���Ĳ�ѯ
        void singlePointQuery(const string &out_file, const Vector3d &point);

        std::vector<int>
        multiPointQuery(const std::vector <Vector3d> &points, double &time, const int &session,
                        const test_type &choice = Test::CPU);

        void multiPointQuery(const string &out_file, const std::vector <Vector3d> &points);

        void multiPointQuery(const string &out_file, const MatrixXd &points);

        void moveOnSurface(const Vector3d &modelVert, const Vector3d &v, const size_t &max_move_cnt);

        void launchParticleSystem(const int &maxIterations, const std::string &out_file);
    };

}
NAMESPACE_END(ITS)