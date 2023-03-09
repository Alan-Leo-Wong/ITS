//#include "CollisionDetection.h"
#include "MyOctree.h"

int main()
{
	/*int s1, s2;
	cout << "resolution of the first model is: ";
	std::cin >> s1;
	cout << "resolution of the second model is: ";
	std::cin >> s2;

	string modelName1 = "bunny";
	string modelName2 = "newbunny";

	CollisionDetection d;
	d.ExtractIntersectLines(s1, s2, modelName1, modelName2, ".off", ".obj");*/

	MyOctree octree(8, "bunny");
	octree.ReadFile("./model/bunny.off");
	octree.createOctree();
	octree.cpIntersection();
	
	return 0;
}
