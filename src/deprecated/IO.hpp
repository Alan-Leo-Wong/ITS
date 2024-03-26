#pragma once
#include <fstream>
#include <Eigen/Dense>

namespace gvis
{
	// Helper function to write single vertex to OBJ file
	static void write_vertex(std::ofstream& output, const Eigen::Vector3d& v)
	{
		output << "v " << v.x() << " " << v.y() << " " << v.z() << std::endl;
	}

	// Helper function to write single vertex to OBJ file
	static void write_vertex(std::ofstream& output, const Eigen::Vector3d& v, const Eigen::Vector3d& rgb)
	{
		output << "v " << v.x() << " " << v.y() << " " << v.z() << " " << rgb.x() << " " << rgb.y() << " " << rgb.z() << std::endl;
	}

	// Helper function to write single vertex to OBJ file
	static void write_vertex_to_xyz(std::ofstream& output, const Eigen::Vector3d& v)
	{
		output << v.x() << " " << v.y() << " " << v.z() << std::endl;
	}

	// Helper function to write face
	static void write_face(std::ofstream& output, const Eigen::Vector3i& f)
	{
		output << "f " << f.x() << " " << f.y() << " " << f.z() << std::endl;
	}

	// Helper function to write face
	static void write_face(std::ofstream& output, const Eigen::Vector4i& f)
	{
		output << "f " << f.x() << " " << f.y() << " " << f.z() << " " << f.w() << std::endl;
	}

	// Helper function to write line
	static void write_line(std::ofstream& output, const Eigen::Vector2i& l)
	{
		output << "l " << l.x() << " " << l.y() << std::endl;
	}

	// Helper function to write full cube (using relative vertex positions in the OBJ file - support for this should be widespread by now)
	inline void writeCube(const Eigen::Vector3d& nodeOrigin, const Eigen::Vector3d& unit, std::ofstream& output, size_t& faceBegIdx)
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
#if defined(MESH_WRITE)
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
#  elif defined(CUBE_WRITE)
		// back
		write_face(output, Eigen::Vector4i(faceBegIdx + 3, faceBegIdx + 4, faceBegIdx + 2, faceBegIdx + 1));
		// bottom
		write_face(output, Eigen::Vector4i(faceBegIdx + 6, faceBegIdx + 5, faceBegIdx + 4, faceBegIdx + 3));
		// right
		write_face(output, Eigen::Vector4i(faceBegIdx + 1, faceBegIdx + 8, faceBegIdx + 6, faceBegIdx + 3));
		// top
		write_face(output, Eigen::Vector4i(faceBegIdx + 1, faceBegIdx + 2, faceBegIdx + 7, faceBegIdx + 8));
		// left
		write_face(output, Eigen::Vector4i(faceBegIdx + 4, faceBegIdx + 5, faceBegIdx + 7, faceBegIdx + 2));
		// front
		write_face(output, Eigen::Vector4i(faceBegIdx + 8, faceBegIdx + 7, faceBegIdx + 5, faceBegIdx + 6));
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

	inline void writePointCloud(const std::vector<Eigen::Vector3d>& points, std::ofstream& output)
	{
		for (size_t i = 0; i < points.size(); ++i)
			write_vertex(output, points[i]);
	}

	inline void writePointCloud_xyz(const std::vector<Eigen::Vector3d>& points, std::ofstream& output)
	{
		for (size_t i = 0; i < points.size(); ++i)
			write_vertex_to_xyz(output, points[i]);
	}

	inline void writePointCloud(const std::vector<Eigen::Vector3d>& points, const std::vector<Eigen::Vector3d>& rgbs, std::ofstream& output)
	{
		for (size_t i = 0; i < points.size(); ++i)
			write_vertex(output, points[i], rgbs[i]);
	}

	inline void writePointCloud(const MXd& points, std::ofstream& output)
	{
		for (size_t i = 0; i < points.size(); ++i)
			write_vertex(output, points.row(i));
	}

	inline void writePointCloud_xyz(const MXd& points, std::ofstream& output)
	{
		for (size_t i = 0; i < points.rows(); ++i)
			write_vertex_to_xyz(output, points.row(i));
	}

	inline void writePointCloud(const MXd& points, const std::vector<Eigen::Vector3d>& rgbs, std::ofstream& output)
	{
		for (size_t i = 0; i < points.rows(); ++i)
			write_vertex(output, points.row(i), rgbs[i]);
	}

	inline void writePointCloud(const Eigen::Vector3d& point, const Eigen::Vector3d& rgb, std::ofstream& output)
	{
		write_vertex(output, point, rgb);
	}
}