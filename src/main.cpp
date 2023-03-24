//#include "CollisionDetection.h"
#include "Octree.h"
#include "utils/Timer.hpp"
#include "utils/String.hpp"
#include "utils/common.hpp"
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

	TimerInterface* timer = nullptr;
	createTimer(&timer);

	startTimer(&timer);

	string modelName = getFileName("", "bunny.off");
	const int maxDepth = 3;
	const int res = 60;
	Octree octree(maxDepth, concatFilePath((string)MODEL_DIR, (string)"bunny.off"));

	stopTimer(&timer);
	double time = getElapsedTime(&timer) * 1e-3;
	printf("Create octree spent %lf s.\n", time);

	startTimer(&timer);
	octree.cpIntersection();
	stopTimer(&timer);
	time = getElapsedTime(&timer) * 1e-3;
	printf("Compute intersection spent %lf s.\n", time);

	startTimer(&timer);
	octree.setSDF();
	stopTimer(&timer);
	time = getElapsedTime(&timer) * 1e-3;
	printf("Compute SDF spent %lf s.\n", time);

	startTimer(&timer);
	octree.cpCoefficients();
	stopTimer(&timer);
	time = getElapsedTime(&timer) * 1e-3;
	printf("Compute coefficients spent %lf s.\n", time);

	startTimer(&timer);
	octree.setBSplineValue();
	stopTimer(&timer);
	time = getElapsedTime(&timer) * 1e-3;
	printf("Compute B-spline value spent %lf s.\n", time);

	octree.textureVisualization(concatFilePath((string)VIS_DIR, modelName, std::to_string(maxDepth), (string)"txt_shell.obj"));
	
	startTimer(&timer);
	octree.mcVisualization(concatFilePath((string)VIS_DIR, modelName, std::to_string(maxDepth), (string)"mc_shell.obj"), V3i(res, res, res));
	stopTimer(&timer);
	time = getElapsedTime(&timer) * 1e-3;
	printf("MarchingCubes spent %lf s.\n", time);

	deleteTimer(&timer);

	return 0;
}
