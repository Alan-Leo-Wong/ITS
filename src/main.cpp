//#include "CollisionDetection.h"
#include "SDFHelper.h"
#include "ThinShells.h"
#include "utils/IO.hpp"
#include "utils/Timer.hpp"
#include "utils/String.hpp"
#include "utils/Common.hpp"
#include "utils/CMDParser.hpp"

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

void testPointInOut(ThinShells& thinShell, const size_t& numPoints, const string& queryFile, const string& queryResFile)
{
	printf("\n[Test] Point INSIDE or OUTSIDE surface\n");

	printf("-- Generate random points in Gaussian Distribution...\n");
	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> randomPointsMat =
		thinShell.generateGaussianRandomPoints(queryFile, numPoints, 8, 0);
	vector<V3d> randomPointsVec(numPoints);
	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>::Map(randomPointsVec.data()->data(), numPoints, 3) = randomPointsMat;

	// timer
	TimerInterface* timer = nullptr;
	createTimer(&timer);

	// ours(cpu/cpu-simd/cuda)
	double time;
	vector<int> our_res = thinShell.multiPointQuery(randomPointsVec, time, Test::CPU);
	if (!our_res.empty()) printf("-- [Ours]: Multi points query spent %lf s.\n", time);
	else return;

	// fcpw
	fcpw::Scene<3> scene;
	fcpw_helper::initSDF(scene, thinShell.getModelVerts(), thinShell.getModelFaces());
	vector<int> fcpw_res(numPoints);
	startTimer(&timer);
	for (size_t i = 0; i < numPoints; ++i)
	{
		double sdf = fcpw_helper::getSignedDistance(randomPointsMat.row(i), scene);
		if (sdf > 0) fcpw_res[i] = 1;
		else if (sdf < 0) fcpw_res[i] = -1;
		else fcpw_res[i] = 0;
	}
	stopTimer(&timer);
	time = getElapsedTime(&timer) * 1e-3;
	printf("-- [FCPW(normal)]: Multi points query spent %lf s.\n", time);

	deleteTimer(&timer);

	// compare result
	size_t correct = 0;
	for (size_t i = 0; i < numPoints; ++i)
		if (our_res[i] == fcpw_res[i]) ++correct;
	//printf("-- Correct number = %llu\n", correct);
	printf("-- Correct rate = %lf%%\n", (correct * 100.0) / numPoints);
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

	startTimer(&timer);
	ThinShells thinShell(concatFilePath((string)MODEL_DIR, (string)"bunny.off"), 16, 16, 16);
	//bool is2Cube = true;
	//ThinShells thinShell(concatFilePath((string)MODEL_DIR, (string)"bunny.off"), 64, 64, 64, is2Cube, 1.0); // to unit cube
	thinShell.creatShell();
	stopTimer(&timer);
	double time = getElapsedTime(&timer) * 1e-3;
	printf("\nCreate shells spent %lf s.\n", time);

	const int treeDepth = thinShell.treeDepth;
	const std::string uniformDir = thinShell.uniformDir;

	startTimer(&timer);
	thinShell.textureVisualization(concatFilePath((string)VIS_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"txt_shell.obj"));
	stopTimer(&timer);
	time = getElapsedTime(&timer) * 1e-3;
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

	const string queryFile = concatFilePath((string)VIS_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"query_point.xyz");
	const string queryResFile = concatFilePath((string)VIS_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"query_point_result.xyz");
	testPointInOut(thinShell, 5000000, queryFile, queryResFile);

	//thinShell.moveOnSurface(V3d(-0.0139834, 0.12456, 0.0302671), V3d(-1e-3, 1e-3, -1e-3), 3);

	deleteTimer(&timer);

	return 0;
}
