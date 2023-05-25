//#include "CollisionDetection.h"
#include "SDFHelper.h"
#include "ThinShells.h"
#include "utils/IO.hpp"
#include "utils/Timer.hpp"
#include "utils/String.hpp"
#include "utils/Common.hpp"
#include "utils/CMDParser.hpp"
#include <igl/signed_distance.h>
#include "TestAllTime.h"

namespace test_time
{
	double test_allTime = .0;
}

std::tuple<UINT, UINT, const char*, const char*> execArgParser(int argc, char** argv)
{
	CmdLineParameter<UINT> _dep_1("d1"); // max depth of the first model's octree
	CmdLineParameter<UINT> _dep_2("d2"); // max depth of the second model's octree
	CmdLineParameter<char*> _mp_1("mp1"); // path of the first model
	CmdLineParameter<char*> _mp_2("mp2"); // path of the second model
	ParserInterface* needParams[] = {
		&_dep_1, &_dep_2, &_mp_1, &_mp_2 };
	if (argc > 1)  cmdLineParse(argc - 1, &argv[1], needParams);

	const unsigned int dep_1 = _dep_1.set ? _dep_1.value : 8;
	const unsigned int dep_2 = _dep_2.set ? _dep_2.value : 8;
	const char* mp_1 = _mp_1.set ? _mp_1.value : "";
	const char* mp_2 = _mp_2.set ? _mp_2.value : "";

	if (mp_1 == "" && mp_2 == "")
	{
		fprintf(stderr, "Please input the path of model! Example: --mp1 \"path\"\n");
		exit(EMPTY_ARG);
	}

	printf("--Octree's max depth built for the FIRST model: %u\n", dep_1);
	printf("--Octree's max depth built for the SECOND model: %u\n", dep_2);

	return std::make_tuple(dep_1, dep_2, mp_1, mp_2);
}

void testPointInOut(ThinShells& thinShell, const size_t& numPoints, const string& queryFile, const string& queryResFile, const int& query_distance)
{
	printf("\n[Test] Point INSIDE or OUTSIDE surface\n");

	printf("-- Generate '%llu' random points in Gaussian Distribution...\n", numPoints);
	vector<V3d> randomPointsVec = thinShell.generateUniformRandomPoints(queryFile, numPoints, query_distance, V3d(0, 0, 0));

	/*MXd mv(randomPointsVec.size(), 3);
	int i = 0;
	for (auto v : randomPointsVec)
	{
		mv.row(i) = v;
		i++;
	}*/
	// timer
	TimerInterface* timer = nullptr;
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
	// 预热
	for (size_t i = 0; i < numPoints; ++i)
	{
		double sdf = fcpw_helper::getSignedDistance(randomPointsVec[i], scene);
		if (sdf > 1e-10) fcpw_res[i] = 1;
		else if (sdf < -1e-10) fcpw_res[i] = -1;
		else fcpw_res[i] = 0;
	}
	// 开始测试
	for (int k = 0; k < 1; ++k)
	{
		printf("-- [FCPW] [Session: %d/%d]", k + 1, session);
		if (k != session - 1) printf("\r");
		else printf("\n");

		startTimer(&timer);
		for (size_t i = 0; i < numPoints; ++i)
		{
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
	vector<V3d> errorPointsVec;
	for (size_t i = 0; i < numPoints; ++i)
	{
		if (our_res[i] == fcpw_res[i]) ++correct;
		else errorPointsVec.emplace_back(randomPointsVec[i]);
	}
	//printf("-- Correct number = %llu\n", correct);
	printf("-- Correct rate = %lf%%\n", (correct * 100.0) / numPoints);

	// 对错误点再调用fcpw
	const size_t numErrorPoints = errorPointsVec.size();
	for (int k = 0; k < session; ++k)
	{
		startTimer(&timer);
		for (size_t i = 0; i < numErrorPoints; ++i)
			fcpw_helper::getSignedDistance(errorPointsVec[i], scene);
		stopTimer(&timer);
	}
	double time_3 = getAverageTimerValue(&timer) * 1e-3;
	printf("-- [Ours+FCPW]: Multi points query spent %lf us.\n", (time_1 + time_3) * 1e6 / numPoints);

	deleteTimer(&timer);
}

int main(int argc, char** argv)
{
	//auto [dep_1, dep_2, mp_1, mp_2] = execArgParser(argc, argv);

	/*int s1, s2;
	cout << "resolution of the first model is: ";
	std::cin >> s1;
	cout << "resolution of the second model is: ";
	std::cin >> s2;

	string modelName1 = "bunny";
	string modelName2 = "newbunny";

	CollisionDetection d;
	d.ExtractIntersectLines(s1, s2, modelName1, modelName2, ".off", ".obj");*/

	cout << "***************************************************\n";
	cout << "**                                               **\n";
	cout << "**            Generate 3D Thin Shells            **\n";
	cout << "**                                               **\n";
	cout << "***************************************************\n";

	string modelName = getFileName("", "bunny.off");
	//const double alpha = 1000;
	cout << "-- Model: " << modelName << endl;
	//cout << "-- alpha: " << alpha << endl;

	TimerInterface* timer = nullptr;
	createTimer(&timer);

	int svo_res = 256;
	//ThinShells thinShell(concatFilePath((string)MODEL_DIR, (string)"dan-m-crowdproject.stl"), svo_res, svo_res, svo_res);

	bool is2Cube = true, isAddNoise = false;
	double noisePercentage = 0.0025;
	//ThinShells thinShell(concatFilePath((string)MODEL_DIR, (string)"Octocat.stl"), svo_res, svo_res, svo_res, is2Cube, 1.0, isAddNoise, noisePercentage); // to unit cube
	ThinShells thinShell(concatFilePath((string)MODEL_DIR, (string)"bunny.off"), svo_res, svo_res, svo_res, is2Cube, 1.0); // to unit cube
	thinShell.creatShell();
	/*stopTimer(&timer);
	double time = getElapsedTime(&timer) * 1e-3;*/
	//printf("\nCreate shells spent %lf s.\n", time);
	printf("\nCreate shells spent %.4lf s.\n", test_time::test_allTime);

	const int treeDepth = thinShell.treeDepth;
	const std::string uniformDir = thinShell.uniformDir;

	startTimer(&timer);
	thinShell.textureVisualization(concatFilePath((string)VIS_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"txt_shell.obj"));
	stopTimer(&timer);
	double time = getElapsedTime(&timer) * 1e-3;
	printf("\nTexture Visualization spent %lf s.\n", time);

	/*const int res = 200;
	const string innerShellFile = concatFilePath((string)VIS_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"mc_innerShell.obj");
	const string outerShellFile = concatFilePath((string)VIS_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"mc_outerShell.obj");
	const string isosurfaceFile = concatFilePath((string)VIS_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"mc_isosurface.obj");
	startTimer(&timer);
	thinShell.mcVisualization(
		innerShellFile, V3i(res, res, res),
		outerShellFile, V3i(res, res, res),
		isosurfaceFile, V3i(res, res, res)
	);
	stopTimer(&timer);
	time = getElapsedTime(&timer) * 1e-3;
	printf("\nMarchingCubes spent %lf s.\n", time);*/

	/*const string queryFile = concatFilePath((string)VIS_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"query_point.xyz");
	const string queryResFile = concatFilePath((string)VIS_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"query_point_result.xyz");
	for(int i = 1; i <= 10; i++)
		testPointInOut(thinShell, 10000, queryFile, queryResFile, i);*/

	/*const string particleResFile = concatFilePath((string)VIS_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"optimized_particle.xyz");
	thinShell.launchParticleSystem(13, particleResFile);*/

	//thinShell.moveOnSurface(V3d(-0.0139834, 0.12456, 0.0302671), V3d(-1e-3, 1e-3, -1e-3), 3);

	deleteTimer(&timer);

	return 0;
}
