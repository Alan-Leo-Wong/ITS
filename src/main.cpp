//#include "CollisionDetection.h"
#include "ThinShells.h"
#include "utils\IO.hpp"
#include "utils\Timer.hpp"
#include "utils\String.hpp"
#include "utils\Common.hpp"
#include "utils\CMDParser.hpp"

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

	string modelName = getFileName("", "switchmec.obj");
	//const double alpha = 1000;
	cout << "-- Model: " << modelName << endl;
	//cout << "-- alpha: " << alpha << endl;

	TimerInterface* timer = nullptr;
	createTimer(&timer);

	startTimer(&timer);
	ThinShells thinShell(concatFilePath((string)MODEL_DIR, (string)"switchmec.obj"), 512, 512, 512);
	thinShell.creatShell();
	stopTimer(&timer);
	double time = getElapsedTime(&timer) * 1e-3;
	printf("\nCreate shells spent %lf s.\n", time);

	const int treeDepth = thinShell.treeDepth;

	startTimer(&timer);
	thinShell.textureVisualization(concatFilePath((string)VIS_DIR, modelName, std::to_string(treeDepth), (string)"txt_shell.obj"));
	stopTimer(&timer);
	time = getElapsedTime(&timer) * 1e-3;
	printf("\nTexture Visualization spent %lf s.\n", time);

	const int res = 200;
	const string innerShellFile = concatFilePath((string)VIS_DIR, modelName, std::to_string(treeDepth), (string)"mc_innerShell.obj");
	const string outerShellFile = concatFilePath((string)VIS_DIR, modelName, std::to_string(treeDepth), (string)"mc_outerShell.obj");
	const string isosurfaceFile = concatFilePath((string)VIS_DIR, modelName, std::to_string(treeDepth), (string)"mc_isosurface.obj");
	startTimer(&timer);
	thinShell.mcVisualization(
		innerShellFile, V3i(res, res, res),
		outerShellFile, V3i(res, res, res),
		isosurfaceFile, V3i(res, res, res)
	);
	stopTimer(&timer);
	time = getElapsedTime(&timer) * 1e-3;
	printf("\nMarchingCubes spent %lf s.\n", time);

	/*const string queryFile = concatFilePath((string)VIS_DIR, modelName, std::to_string(treeDepth), (string)"query_point.obj");
	MXd randomPoints = thinShell.generateRandomPoints(queryFile, 100);

	startTimer(&timer);
	const string queryResFile = concatFilePath((string)VIS_DIR, modelName, std::to_string(treeDepth), (string)"query_point_result.obj");
	thinShell.multiPointQuery(queryResFile, randomPoints);
	stopTimer(&timer);
	time = getElapsedTime(&timer) * 1e-3;
	printf("\nMulti points query spent %lf s.\n", time);*/

	deleteTimer(&timer);

	return 0;
}
