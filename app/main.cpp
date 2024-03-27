#include "TestAllTime.h"

#ifdef FCPW_TEST

#include "FCPWHelper.hpp"

#endif

#include "core/ThinShells.hpp"
#include "detail/Geometry.hpp"
#include "utils/Timer.hpp"
#include "utils/File.hpp"
#include "utils/Common.hpp"
#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>
#include <igl/signed_distance.h>

using namespace ITS;
using namespace ITS::core;
using namespace ITS::utils::timer;
using namespace ITS::utils::file;
using namespace Eigen;

namespace test_time {
    double test_allTime = .0;
}

void testPointInOut(ThinShells &thinShell, size_t numPoints,
                    const string &queryFile, const string &queryResFile,
                    int queryDistance) {
    logger().info("\n[Test] Point INSIDE or OUTSIDE the surface");

    printf("-- Generate '%llu' random points in Uniform Distribution...", numPoints);

    const auto bbox = thinShell.getBoundingBox();
    Eigen::Matrix<double, Eigen::Dynamic, 3> randomPoints;
    pointgen::genUniformRandomPoints(bbox, numPoints, randomPoints, queryDistance);
    vector<Vector3d> randomPointsVec;
    utils::matrix2List(randomPoints, randomPointsVec);

    /*MXd mv(randomPointsVec.size(), 3);
    int i = 0;
    for (auto v : randomPointsVec)
    {
        mv.row(i) = v;
        i++;
    }*/
    // timer
    TimerInterface *timer = nullptr;
    createTimer(&timer);

    // ours(cpu/cpu-simd/cuda)
    //VXi I;
    //MXd C, N;
    //VXd iglres;
    //igl::signed_distance(mv, thinShell.getVertices(), thinShell.getFaces(),
    //	igl::SignedDistanceType::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER, iglres, I, C, N);

    int session = 3;
    double time_1;
    auto testChoice = Test::CPU;
    vector<int> our_res = thinShell.multiPointQuery(randomPointsVec, time_1, session, testChoice);
    if (!our_res.empty()) printf("-- [Ours]: Multi points query spent %lf us.\n", time_1 * 1e6 / numPoints);
    else return;

    // fcpw
    fcpw::Scene<3> scene;
    fcpw_helper::initSDF(scene, thinShell.getModelVerts(), thinShell.getModelFaces());
    vector<int> fcpw_res(numPoints);
    for (size_t i = 0; i < numPoints; ++i) {
        double sdf = fcpw_helper::getSignedDistance(randomPointsVec[i], scene);
        if (sdf > 1e-10) fcpw_res[i] = 1;
        else if (sdf < -1e-10) fcpw_res[i] = -1;
        else fcpw_res[i] = 0;
    }
    for (int k = 0; k < 1; ++k) {
        printf("-- [FCPW] [Session: %d/%d]", k + 1, session);
        if (k != session - 1) printf("\r");
        else printf("\n");

        startTimer(&timer);
        for (size_t i = 0; i < numPoints; ++i) {
            double sdf = fcpw_helper::getSignedDistance(randomPointsVec[i], scene);
            if (sdf > 0) fcpw_res[i] = 1;
            else if (sdf < 0) fcpw_res[i] = -1;
            else fcpw_res[i] = 0;
        }
        stopTimer(&timer);
    }
    double time_2 = getAverageTimerValue(&timer) * 1e-3;
    printf("-- [FCPW(normal)]: Multi points query spent %lf us.\n", time_2 * 1e6 / numPoints);

    resetTimer(&timer);

    // compare result
    size_t correct = 0;
    vector<Vector3d> errorPointsVec;
    for (size_t i = 0; i < numPoints; ++i) {
        if (our_res[i] == fcpw_res[i]) ++correct;
        else errorPointsVec.emplace_back(randomPointsVec[i]);
    }
    //printf("-- Correct number = %llu\n", correct);
    printf("-- Correct rate = %lf%%\n", (correct * 100.0) / numPoints);

    const size_t numErrorPoints = errorPointsVec.size();
    for (int k = 0; k < session; ++k) {
        startTimer(&timer);
        for (size_t i = 0; i < numErrorPoints; ++i)
            fcpw_helper::getSignedDistance(errorPointsVec[i], scene);
        stopTimer(&timer);
    }
    double time_3 = getAverageTimerValue(&timer) * 1e-3;
    printf("-- [Ours+FCPW]: Multi points query spent %lf us.\n", (time_1 + time_3) * 1e6 / numPoints);

    deleteTimer(&timer);
}

int main(int argc, char **argv) {
    std::cout << "***************************************************\n";
    std::cout << "**                                               **\n";
    std::cout << "**        Generate 3D Implicit Thin Shells       **\n";
    std::cout << "**                                               **\n";
    std::cout << "***************************************************\n";

    ////////////////////////////////////////////////////////

    struct {
        std::string meshFile;
        bool meshNorm = false;
        bool addNoise = false;
        double noisePercentage = -1.0;

        int svoRes;

        bool mcVis = false;
        int mcRes = -1;
        std::string mcOutDir;
    } args;
    CLI::App app{"3D Implicit Thin Shells"};

    app.add_option("-f,--in_file", args.meshFile, "Input mesh file")->required();

    app.add_flag("-U,--mesh_norm", args.meshNorm, "Normalize the input mesh");

    app.add_flag("-N,--mesh_noise", args.addNoise, "Add noise on the mesh");
    app.add_option("-P,--mesh_noise_per", args.noisePercentage, "Noise percentage for adding noise on the mesh")
            ->check(CLI::Validator(
                    std::function < std::string(std::string & ) > ([&args](std::string &val) -> std::string {
                        if (args.addNoise && std::stod(val) < .0)
                            throw CLI::ValidationError(
                                    "If mesh noise is enabled, the noise percentage must be provided and larger than zero");
                        return "";
                    }), "noise check"));

    app.add_option("-r,--svo_res", args.svoRes, "Resolution of sparse voxel octree")->required();

    app.add_flag("-M,--mc", args.mcVis, "Perform marching-cubes visualization");
    app.add_option("-R,--mc_res", args.mcRes, "Resolution of marching-cubes")
            ->check(CLI::Validator(
                    std::function < std::string(std::string & ) > ([&args](std::string &val) -> std::string {
                        if (args.mcVis && std::stoi(val) < 0)
                            throw CLI::ValidationError(
                                    "If marching-cubes is enabled, the resolution must be provided and larger than zero");
                        return "";
                    }), "mc res check"));
    app.add_option("-O,--mc_dir", args.mcOutDir, "Output directory of the shells via marching cubes")
            ->check(CLI::Validator(
                    std::function < std::string(std::string & ) > ([&args](std::string &val) -> std::string {
                        if (args.mcVis && val.empty())
                            throw CLI::ValidationError(
                                    "If marching-cubes is enabled, the output directory must be provided");
                        return "";
                    }), "mc out check"));

    try {
        app.parse(argc, argv);
    }
    catch (const CLI::ValidationError &e) {
        return app.exit(e);
    }
    catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    ////////////////////////////////////////////////////////

    string modelName = getFileName(args.meshFile);
    logger().info("Model: {}", modelName);

    TimerInterface *timer = nullptr;
    createTimer(&timer);

    bool lazyTag = (args.meshNorm || args.addNoise);
    ThinShells thinShell(args.meshFile, args.svoRes, lazyTag); // to unit cube
    if (args.meshNorm) thinShell.model2UnitCube();
    if (args.meshNorm) thinShell.addNoise(args.noisePercentage);

    thinShell.creatShell();
    logger().info("Generate shells spent {} s.", test_time::test_allTime);

    int treeDepth = thinShell.treeDepth;
    std::string uniformDir = thinShell.uniformDir;

    /*startTimer(&timer);
    thinShell.textureVisualization(
            concatFilePath(VIS_DIR, modelName, uniformDir, std::to_string(treeDepth), "txt_shell.obj"));
    stopTimer(&timer);
    double time = getElapsedTime(&timer) * 1e-3;
    printf("\nTexture Visualization spent %lf s.\n", time);*/

    if (args.mcVis) {
        utils::split();

        logger().info("The resolution of MarchingCubes is {}.", args.mcRes);

        string innerShellFile = concatFilePath(args.mcOutDir, modelName, uniformDir,
                                               std::to_string(treeDepth),
                                               "mc_innerShell.obj");
        string outerShellFile = concatFilePath(args.mcOutDir, modelName, uniformDir,
                                               std::to_string(treeDepth),
                                               "mc_outerShell.obj");
        string isosurfaceFile = concatFilePath(args.mcOutDir, modelName, uniformDir,
                                               std::to_string(treeDepth),
                                               "mc_isosurface.obj");
        startTimer(&timer);
        thinShell.mcVisualization(
                innerShellFile, Vector3i(args.mcRes, args.mcRes, args.mcRes),
                outerShellFile, Vector3i(args.mcRes, args.mcRes, args.mcRes),
                isosurfaceFile, Vector3i(args.mcRes, args.mcRes, args.mcRes)
        );
        stopTimer(&timer);
        double time = getElapsedTime(&timer) * 1e-3;
        logger().info("MarchingCubes spent {} s.", time);
    }

    /*const string queryFile = concatFilePath((string)VIS_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"query_point.xyz");
    const string queryResFile = concatFilePath((string)VIS_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"query_point_result.xyz");
    for(int i = 1; i <= 10; i++)
        testPointInOut(thinShell, 10000, queryFile, queryResFile, i);*/

    deleteTimer(&timer);

    return 0;
}
