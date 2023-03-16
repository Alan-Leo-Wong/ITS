#pragma once
#include "BasicDataType.h"
//#define FCPW_USE_ENOKI
//#define FCPW_SIMD_WIDTH 4
#include <fcpw/fcpw.h>

double getSignedDistance(const V3d& queryPoint, const fcpw::Scene<3>& scene)
{
	// perform a closest point query
	fcpw::Interaction<3> interaction;
	scene.findClosestPoint(queryPoint, interaction);
	/*cout << "p = " << interaction.p << endl;
	cout << "d = " << interaction.d << endl;
	cout << "n = " << interaction.n << endl;*/
	//cout << "���� = " << interaction.signedDistance(queryPoint) << endl;
	return interaction.signedDistance(queryPoint);
}

void initSDF(fcpw::Scene<3>& scene, const vector<V3d>& modelVerts, const vector<V3i>& modelFaces)
{
	// set the types of primitives the objects in the scene contain;
	// in this case, we have a single object consisting of only triangles
	scene.setObjectTypes({ {fcpw::PrimitiveType::Triangle} });

	// set the vertex and triangle count of the (0th) object
	UINT nVertices = modelVerts.size();
	UINT nTriangles = modelFaces.size();
	scene.setObjectVertexCount(nVertices, 0);
	scene.setObjectTriangleCount(nTriangles, 0);

	// specify the vertex positions
	for (int i = 0; i < nVertices; i++)
		scene.setObjectVertex(modelVerts[i], i, 0);

	// specify the triangle indices
	for (int i = 0; i < nTriangles; i++)
		scene.setObjectTriangle(modelFaces[i].data(), i, 0);

	scene.computeObjectNormals(0);

	// now that the geometry has been specified, build the acceleration structure
	scene.build(fcpw::AggregateType::Bvh_SurfaceArea, true); // the second boolean argument enables vectorization
}