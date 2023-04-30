#pragma once
#include <fstream>
#include <Eigen\dense>
#include "cuda\CUDAMacro.h"

// An Axis Aligned Box (AAB) of a certain type - to be initialized with a boxOrigin and boxEnd
template <typename Real>
struct AABox {
	Real boxOrigin;
	Real boxEnd;
	Real boxWidth;

	_CUDA_GENERAL_CALL_ AABox() : boxOrigin(Real()), boxEnd(Real()), boxWidth(Real()) {}
	_CUDA_GENERAL_CALL_ AABox(Real _boxOrigin, Real _max) : boxOrigin(_min), boxEnd(_max), boxWidth(_max - _min) {}
};

template <typename Real>
struct Triangle
{
	Real p1, p2, p3;
	Real normal;
	double area;
	double dir;

	_CUDA_GENERAL_CALL_ Triangle() {}

	_CUDA_GENERAL_CALL_ Triangle(const Real& _p1, const Real& _p2, const Real& _p3) :p1(_p1), p2(_p2), p3(_p3) {}
};

namespace gvis
{
	// Helper function to write single vertex to OBJ file
	static void write_vertex(std::ofstream& output, const Eigen::Vector3d& v) 
	{
		output << "v " << v.x() << " " << v.y() << " " << v.z() << std::endl;
	}

	// Helper function to write face
	static void write_face(std::ofstream& output, const Eigen::Vector3i& f) 
	{
		output << "f " << f.x() << " " << f.y() << " " << f.z() << std::endl;
	}

	// Helper function to write line
	static void write_line(std::ofstream& output, const Eigen::Vector2i& l) 
	{
		output << "l " << l.x() << " " << l.y() << std::endl;
	}

	// Helper function to write full cube (using relative vertex positions in the OBJ file - support for this should be widespread by now)
	void writeCube(const Eigen::Vector3d& nodeOrigin, const Eigen::Vector3d& unit, std::ofstream& output, size_t& faceBegIdx) 
	{
		//	   2-------1
		//	  /|      /|
		//	 / |     / |
		//	7--|----8  |
		//	|  4----|--3
		//	| /     | /
		//	5-------6
		// Create vertices
		Eigen::Vector3d v1 = nodeOrigin + Eigen::Vector3d(0, unit.y(), unit.z());
		Eigen::Vector3d v2 = nodeOrigin + Eigen::Vector3d(0, 0, unit.z());
		Eigen::Vector3d v3 = nodeOrigin + Eigen::Vector3d(0, unit.y(), 0);
		Eigen::Vector3d v4 = nodeOrigin;
		Eigen::Vector3d v5 = nodeOrigin + Eigen::Vector3d(unit.x(), 0, 0);
		Eigen::Vector3d v6 = nodeOrigin + Eigen::Vector3d(unit.x(), unit.y(), 0);
		Eigen::Vector3d v7 = nodeOrigin + Eigen::Vector3d(unit.x(), 0, unit.z());
		Eigen::Vector3d v8 = nodeOrigin + Eigen::Vector3d(unit.x(), unit.y(), unit.z());

		// write them in reverse order, so relative position is -i for v_i
		write_vertex(output, v1);
		write_vertex(output, v2);
		write_vertex(output, v3);
		write_vertex(output, v4);
		write_vertex(output, v5);
		write_vertex(output, v6);
		write_vertex(output, v7);
		write_vertex(output, v8);

		// create faces
#ifdef MESH_WRITE
		// back
		write_face(output, Eigen::Vector3i(faceBegIdx + 1, faceBegIdx + 3, faceBegIdx + 4));
		write_face(output, Eigen::Vector3i(faceBegIdx + 1, faceBegIdx + 4, faceBegIdx + 2));
		// bottom								   
		write_face(output, Eigen::Vector3i(faceBegIdx + 4, faceBegIdx + 3, faceBegIdx + 6));
		write_face(output, Eigen::Vector3i(faceBegIdx + 4, faceBegIdx + 6, faceBegIdx + 5));
		// right								   	 
		write_face(output, Eigen::Vector3i(faceBegIdx + 3, faceBegIdx + 1, faceBegIdx + 8));
		write_face(output, Eigen::Vector3i(faceBegIdx + 3, faceBegIdx + 8, faceBegIdx + 6));
		// top									   	 
		write_face(output, Eigen::Vector3i(faceBegIdx + 1, faceBegIdx + 2, faceBegIdx + 7));
		write_face(output, Eigen::Vector3i(faceBegIdx + 1, faceBegIdx + 7, faceBegIdx + 8));
		// left									   	 
		write_face(output, Eigen::Vector3i(faceBegIdx + 2, faceBegIdx + 4, faceBegIdx + 5));
		write_face(output, Eigen::Vector3i(faceBegIdx + 2, faceBegIdx + 5, faceBegIdx + 7));
		// front								   	  
		write_face(output, Eigen::Vector3i(faceBegIdx + 5, faceBegIdx + 6, faceBegIdx + 8));
		write_face(output, Eigen::Vector3i(faceBegIdx + 5, faceBegIdx + 8, faceBegIdx + 7));
#  else
		write_line(output, Eigen::Vector2i(faceBegIdx + 1, faceBegIdx + 2));
		write_line(output, Eigen::Vector2i(faceBegIdx + 2, faceBegIdx + 7));
		write_line(output, Eigen::Vector2i(faceBegIdx + 7, faceBegIdx + 8));
		write_line(output, Eigen::Vector2i(faceBegIdx + 8, faceBegIdx + 1));

		write_line(output, Eigen::Vector2i(faceBegIdx + 3, faceBegIdx + 4));
		write_line(output, Eigen::Vector2i(faceBegIdx + 4, faceBegIdx + 5));
		write_line(output, Eigen::Vector2i(faceBegIdx + 5, faceBegIdx + 6));
		write_line(output, Eigen::Vector2i(faceBegIdx + 6, faceBegIdx + 3));

		write_line(output, Eigen::Vector2i(faceBegIdx + 3, faceBegIdx + 1));
		write_line(output, Eigen::Vector2i(faceBegIdx + 4, faceBegIdx + 2));
		write_line(output, Eigen::Vector2i(faceBegIdx + 5, faceBegIdx + 7));
		write_line(output, Eigen::Vector2i(faceBegIdx + 6, faceBegIdx + 8));
#endif

		faceBegIdx += 8;
	}
}