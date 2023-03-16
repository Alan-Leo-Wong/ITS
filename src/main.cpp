//#include "CollisionDetection.h"
#include "Octree.h"
#include "utils/common.hpp"
#include "utils/Timer.hpp"
#include "utils/CMDParser.hpp"

//#ifndef MODEL_DIR
//#define MODEL_DIR ".\model\"
//#endif
//#ifndef OUT_DIR
//#define OUT_DIR ".\output\"
//#endif

std::tuple<UINT, UINT, const char*, const char*> execArgParser(int argc, char** argv)
{
	CmdLineParameter<UINT> _res_1("res1"); // resolution of the first model
	CmdLineParameter<UINT> _res_2("res2"); // resolution of the second model
	CmdLineParameter<char*> _mp_1("mp1"); // path of the first model
	CmdLineParameter<char*> _mp_2("mp2"); // path of the second model
	ParserInterface* needParams[] = {
		&_res_1, &_res_2, &_mp_1, &_mp_2 };
	if (argc > 1)  cmdLineParse(argc - 1, &argv[1], needParams);

	const unsigned int res_1 = _res_1.set ? _res_1.value : 10;
	const unsigned int res_2 = _res_2.set ? _res_2.value : 10;
	const char* mp_1 = _mp_1.set ? _mp_1.value : "";
	const char* mp_2 = _mp_2.set ? _mp_2.value : "";

	if (mp_1 == "" || mp_2 == "")
	{
		printf("Please input the path of model! Example: --mp1 \"path\"\n");
		exit(EMPTY_ARG);
	}

	printf("--Resolution of the FIRST model: %u\n", res_1);
	printf("--Resolution of the SECOND model: %u\n", res_2);

	return std::make_tuple(res_1, res_2, mp_1, mp_2);
}

int main(int argc, char** argv)
{
	//auto [res_1, res_2, mp_1, mp_2] = execArgParser(argc, argv);

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
	Octree octree(4, "./model/bunny.off");
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
	octree.setInDomainLeafNode();
	stopTimer(&timer);
	time = getElapsedTime(&timer) * 1e-3;
	printf("Select indomain leaf nodes spent %lf s.\n", time);

	startTimer(&timer);
	octree.setBSplineValue();
	stopTimer(&timer);
	time = getElapsedTime(&timer) * 1e-3;
	printf("Compute B-spline value spent %lf s.\n", time);

	octree.saveBSplineValue("./output/4/BSplineValue.txt");

	startTimer(&timer);
	octree.mcVisualization("./vis/4/mc_shell.obj", V3i(20, 20, 20));
	stopTimer(&timer);
	time = getElapsedTime(&timer) * 1e-3;
	printf("MarchingCubes spent %lf s.\n", time);

	octree.txtVisualization("./vis/4/txt_shell.obj");

	deleteTimer(&timer);

	return 0;
}
