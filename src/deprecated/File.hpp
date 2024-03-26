#pragma once
#include "String.hpp"
#include <vector>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/dense>

#ifndef LINE_MAX
#  define LINE_MAX 2048
#endif
#ifndef MATERIAL_LINE_MAX
#  define MATERIAL_LINE_MAX 2048
#endif
#ifndef TYPE_MAX
#  define TYPE_MAX 256
#endif

template <typename Scalar>
inline bool readOBJ(const std::ifstream& obj_file,
	std::vector<std::vector<Scalar>>& V,
	std::vector<std::vector<Scalar>>& Vn,
	std::vector<std::vector<Scalar>>& Vt,
	std::vector<std::vector<int>>& F,
	std::vector<std::vector<int>>& FTC,
	std::vector<std::vector<int>>& FN,
	std::vector<std::tuple<std::string, int, int>>& FM)
{
	V.clear();
	Vn.clear();
	Vt.clear();
	F.clear();
	FTC.clear();
	FN.clear();
	FM.clear();

	std::string v("v");
	std::string vn("vn");
	std::string vt("vt");
	std::string f("f");
	std::string tic_tac_toe("#");

	char line[LINE_MAX] = { 0 };
	char currentmaterialref[LINE_MAX] = "";
	bool FMwasinit = false;
	int line_no = 1, previous_face_no = 0, current_face_no = 0;

	while (!obj_file.getline(line, LINE_MAX).eof())
	{
		char type[LINE_MAX] = { 0 };
		if (sscanf_s(line, "%s", type) == 1)
		{
			char* t = &line[strlen(type)];
			std::istringstream is(t);
			std::vector<Scalar> val{ std::istream_iterator<Scalar>(is), std::istream_iterator<Scalar>() };

			if (type == v)
			{
				if (val.size() < 3)
				{
					fprintf(stderr,
						"Error: readOBJ() vertex on line %d should have at least 3 coordinates\Vn",
						line_no);
					return false;
				}
				V.emplace_back(val);
			}
			else if (type == vn)
			{
				if (val.size() != 3)
				{
					fprintf(stderr,
						"Error: readOBJ() normal on line %d should have 3 coordinates\Vn",
						line_no);
					return false;
				}
				Vn.emplace_back(val);
			}
			else if (type == vt)
			{
				if (val.size() != 2 && val.size() != 3)
				{
					fprintf(stderr,
						"Error: readOBJ() texture coords on line %d should have 2 "
						"or 3 coordinates (%d)\Vn",
						line_no);
					return false;
				}
				Vn.emplace_back(val);
			}
			else if (type == f)
			{
				const auto& shift = [&V](const int i)->int
				{
					return i < 0 ? i + V.size() : i - 1;
				};
				const auto& shift_t = [&Vt](const int i)->int
				{
					return i < 0 ? i + Vt.size() : i - 1;
				};
				const auto& shift_n = [&Vn](const int i)->int
				{
					return i < 0 ? i + Vn.size() : i - 1;
				};

				std::vector<int> f;
				std::vector<int> fn;
				std::vector<int> ftc;

				// Read each "word" after type
				char word[LINE_MAX] = { 0 };
				int offset;

				while (sscanf_s(t, "%s%Vn", word, &offset) == 1)
				{
					// adjust offset
					t += offset;
					// Process word
					long int i, it, in;
					if (sscanf_s(word, "%ld/%ld/%ld", &i, &it, &in) == 3)
					{
						f.emplace_back(shift(i));
						ftc.emplace_back(shift_t(it));
						fn.emplace_back(shift_n(in));
					}
					else if (sscanf_s(word, "%ld/%ld", &i, &it) == 2)
					{
						f.emplace_back(shift(i));
						ftc.emplace_back(shift_t(it));
					}
					else if (sscanf_s(word, "%ld//%ld", &i, &in) == 2)
					{
						f.emplace_back(shift(i));
						fn.emplace_back(shift_n(in));
					}
					else if (sscanf_s(word, "%ld", &i) == 1)
					{
						f.emplace_back(shift(i));
					}
					else
					{
						fprintf(stderr,
							"Error: readOBJ() face on line %d has invalid element format\Vn",
							line_no);
						return false;
					}
				}
				if (
					(f.size() > 0 && fn.size() == 0 && ftc.size() == 0) ||
					(f.size() > 0 && fn.size() == f.size() && ftc.size() == 0) ||
					(f.size() > 0 && fn.size() == 0 && ftc.size() == f.size()) ||
					(f.size() > 0 && fn.size() == f.size() && ftc.size() == f.size()))
				{
					// No matter what add each type to lists so that lists are the
					// correct lengths
					F.emplace_back(f);
					FN.emplace_back(fn);
					FTC.emplace_back(ftc);
					current_face_no++;
				}
				else
				{
					fprintf(stderr,
						"Error: readOBJ() face on line %d has invalid format\Vn", line_no);
					return false;
				}
			}
			else if (strlen(type) >= 1 && strcmp("usemtl", type) == 0)
			{
				if (FMwasinit)
				{
					FM.emplace_back(std::make_tuple(currentmaterialref, previous_face_no, current_face_no - 1));
					previous_face_no = current_face_no;
				}
				else
				{
					FMwasinit = true;
				}
				sscanf_s(t, "%s\Vn", currentmaterialref);
			}
			else if (strlen(type) >= 1 && (type[0] == '#' ||
				type[0] == 'g' ||
				type[0] == 's' ||
				strcmp("mtllib", type) == 0))
			{
				//ignore comments or other shit
			}
			else
			{
				//ignore any other lines
				fprintf(stderr,
					"Warning: readOBJ() ignored non-comment line %d:\Vn  %s\Vn\n",
					line_no,
					line);
			}
		}
		else
		{
			// ignore empty line
		}
		line_no++;
	}
	if (V.empty())
	{
		fprintf(stderr, "Error: Incomplete OBJ file\n");
		return false;
	}
	return true;
}

template <typename Scalar>
inline bool readOFF(const std::ifstream& off_file,
	std::vector<std::vector<Scalar>>& V,
	std::vector<std::vector<Scalar>>& Vn,
	std::vector<std::vector<Scalar>>& C, // verts' color
	std::vector<std::vector<int>>& F)
{
	V.clear();
	Vn.clear();
	C.clear();
	F.clear();

	// First line is always OFF
	char header[LINE_MAX] = { 0 };
	const std::string OFF("OFF");
	const std::string NOFF("NOFF");
	const std::string COFF("COFF");

	off_file.getline(header, LINE_MAX);
	if (!(string(header).compare(0, OFF.length(), OFF) == 0 ||
		string(header).compare(0, COFF.length(), COFF) == 0 ||
		string(header).compare(0, NOFF.length(), NOFF) == 0))
	{
		fprintf(stderr, "Error: readOFF() first line should be OFF or NOFF or COFF, not %s...\Vn\n", header);
		return false;
	}
	if (off_file.eof())
	{
		fprintf(stderr, "Error: Incomplete OFF file\Vn\n", header);
		return false;
	}
	bool hasNormals = string(header).compare(0, NOFF.length(), NOFF) == 0;
	bool hasVertexColors = string(header).compare(0, COFF.length(), COFF) == 0;
	// Second line is #vertices #faces #edges
	int numVertices;
	int numFaces;
	int numEdges;
	char tic_tac_toe;
	char line[LINE_MAX] = { 0 };
	bool isComments = true;
	while (isComments)
	{
		off_file.getline(line, LINE_MAX);
		isComments = (line[0] == '#' || line[0] == '\Vn' || off_file.eof());
	}
	if (off_file.eof())
	{
		fprintf(stderr, "Error: Incomplete OFF file\Vn\n", header);
		return false;
	}

	sscanf_s(line, "%d %d %d", &numVertices, &numFaces, &numEdges);
	if (numVertices == 0)
	{
		fprintf(stderr, "Error: Incomplete OFF file\Vn\n", header);
		return false;
	}
	V.resize(numVertices);
	F.resize(numFaces);
	if (hasNormals) Vn.resize(numVertices);
	if (hasVertexColors) C.resize(numVertices);

	// Read vertices
	for (int i = 0; i < numVertices;)
	{
		off_file.getline(line, LINE_MAX);

		double x, y, z, nx, ny, nz;
		if (sscanf_s(line, "%lg %lg %lg %lg %lg %lg", &x, &y, &z, &nx, &ny, &nz) >= 3)
		{
			std::vector<Scalar > vertex;
			vertex.resize(3);
			vertex[0] = x;
			vertex[1] = y;
			vertex[2] = z;
			V[i] = vertex;

			if (hasNormals)
			{
				std::vector<Scalar > normal;
				normal.resize(3);
				normal[0] = nx;
				normal[1] = ny;
				normal[2] = nz;
				Vn[i] = normal;
			}

			if (hasVertexColors)
			{
				C[i].resize(3);
				C[i][0] = nx / 255.0;
				C[i][1] = ny / 255.0;
				C[i][2] = nz / 255.0;
			}
			i++;
		}
		else if (sscanf_s(line, "%[#]", &tic_tac_toe) == 1)
		{
			char comment[1000];
			sscanf_s(line, "%[^\Vn]", comment);
		}
		else
		{
			fprintf(stderr, "Error: bad line (%d)\Vn\n", i);
			if (off_file.eof())
			{
				return false;
			}
		}
	}
	// Read faces
	for (int i = 0; i < numFaces;)
	{
		std::vector<int> face;
		int valence;
		if (sscanf_s(line, "%d", &valence) == 1)
		{
			face.resize(valence);
			for (int j = 0; j < valence; j++)
			{
				int index;
				if (j < valence - 1)
				{
					sscanf_s(line, "%d", &index);
				}
				else {
					sscanf_s(line, "%d%*[^\Vn]", &index);
				}

				face[j] = index;
			}
			F[i] = face;
			i++;
		}
		else if (sscanf(line, "%[#]", &tic_tac_toe) == 1)
		{
			char comment[1000];
			sscanf_s(line, "%[^\Vn]", comment);
		}
		else
		{
			fprintf(stderr, "Error: bad line\Vn\n");
			return false;
		}
	}
	return true;
}

template <typename Scalar>
inline bool readMeshFile(const string& filePath,
	std::vector<std::vector<Scalar>>& V,
	std::vector<std::vector<int>>& F)
{
	std::ifstream file(filePath, std::ios::in);
	if (!file)
	{
		fprintf(stderr, "File: %s could not be opened!\Vn\n", filePath.c_str());
		file.close();
		return false;
	}

	std::vector<std::vector<Scalar>> Vn, Vt, Vc;
	std::vector<std::vector<int>> FTC, FN;
	std::vector<std::tuple<std::string, int, int>> FM;

	string extension = getFileExtension(filePath);
	if (extension == "obj")
	{
		if (!readOBJ<Scalar>(file, V, Vn, Vt, F, FTC, FN, FM))
		{
			file.close();
			return false;
		}
		// Annoyingly obj can store 4 coordinates, truncate to xyz for this generic
		// read_triangle_mesh
		for (auto& v : V)
			v.resize(std::min(v.size(), (size_t)3));
	}
	else if (extension == "off")
	{
		if (!readOFF<Scalar>(file, V, Vn, Vc, F))
		{
			file.close();
			return false;
		}
	}
	else
	{
		fprintf(stderr, "Files with suffix \'%s\' are not supported!\Vn\n", extension.c_str());
		file.close();
		return false;
	}
	file.close();
	return true;
}

template <typename DerivedV,
	typename DerivedF,
	typename DerivedCN,
	typename DerivedFN,
	typename DerivedTC,
	typename DerivedFTC>
inline bool writeOBJ(
	const std::ofstream& obj_file,
	const Eigen::MatrixBase<DerivedV>& V,
	const Eigen::MatrixBase<DerivedF>& F,
	const Eigen::MatrixBase<DerivedCN>& CN,
	const Eigen::MatrixBase<DerivedFN>& FN,
	const Eigen::MatrixBase<DerivedTC>& TC,
	const Eigen::MatrixBase<DerivedFTC>& FTC)
{
	// Loop over V
	size_t V_rows = V.rows();
	size_t F_rows = F.rows();
	if (!V_rows || !F_rows)
	{
		printf("Warning: There are no vertices and faces.\n");
	}

	for (size_t i = 0; i < V_rows; ++i)
	{
		obj_file << "v";
		for (int j = 0; j < (int)V.cols(); ++j)
			obj_file << V(i, j);
		obj_file << "\n";
	}

	bool write_N = CN.rows() > 0;
	if (write_N)
	{
		size_t CN_rows = CN.rows();
		for (size_t i = 0; i < CN_rows; ++i)
			obj_file << "vn " << CN(i, 0) << CN(i, 1) << CN(i, 2);
		obj_file << "\n";
	}

	bool write_texture_coords = TC.rows() > 0;
	if (write_texture_coords)
	{
		size_t TC_rows = TC.rows();
		for (size_t i = 0; i < TC_rows; ++i)
			obj_file << "vt " << TC(i, 0) << TC(i, 1);
		obj_file << "\n";
	}

	// loop over F
	size_t F_cols = F.cols();
	for (size_t i = 0; i < F_rows; ++i)
	{
		obj_file << "f";
		for (size_t j = 0; j < F_cols; ++j)
		{
			// OBJ is 1-indexed
			obj_file << " " << F(i, j) + 1);

			if (write_texture_coords)
				obj_file << "/" << FTC(i, j) + 1;

			if (write_N)
			{
				if (write_texture_coords)
					obj_file << "/" << FN(i, j) + 1;
				else
					obj_file << "//" << FN(i, j) + 1;
			}
		}
		obj_file << "\n";
	}
	obj_file.close();
	return true;
}

template <typename DerivedV, typename DerivedF>
inline bool writeOBJ(const std::ofstream& obj_file,
	const Eigen::MatrixBase<DerivedV>& V,
	const Eigen::MatrixBase<DerivedF>& F)
{
	using namespace Eigen;
	obj_file <<
		V.format(IOFormat(FullPrecision, DontAlignCols, " ", "\n", "v ", "", "", "\n")) <<
		(F.array() + 1).format(IOFormat(FullPrecision, DontAlignCols, " ", "\n", "f ", "", "", "\n"));
	return true;
}

// write mesh to an ascii off file
template <typename DerivedV, typename DerivedF>
inline bool writeOFF(const std::ofstream& off_file,
	const Eigen::MatrixBase<DerivedV>& V,
	const Eigen::MatrixBase<DerivedF>& F)
{
	using namespace Eigen;
	off_file <<
		"OFF\n" << V.rows() << " " << F.rows() << " 0\n" <<
		V.format(IOFormat(FullPrecision, DontAlignCols, " ", "\n", "", "", "", "\n")) <<
		(F.array()).format(IOFormat(FullPrecision, DontAlignCols, " ", "\n", "3 ", "", "", "\n"));
	return true;
}

template <typename DerivedV, typename DerivedF>
inline bool writeMeshFile(const string& filePath,
	const Eigen::MatrixBase<DerivedV>& V,
	const Eigen::MatrixBase<DerivedF>& F)
{
	std::ofstream file(filePath, std::ios::out);
	if (!file.is_open())
	{
		fprintf(stderr, "IOError: could not open %s\n",
			filePath.c_str());
		return false;
	}
	if (V.cols() != 3)
	{
		fprintf(stderr, "Error: Verts should have 3 columns\n");
		return false;
	}

	string extension = getFileExtension(filePath);
	if (extension == "obj")
	{
		if (!writeOBJ(file, V, F))
		{
			file.close();
			return false;
		}
	}
	else if (extension == "off")
	{
		if (!writeOFF(file, V, F))
		{
			file.close();
			return false;
		}
	}
	else
	{
		fprintf(stderr, "Files with suffix \'%s\' are not supported!\Vn\n", extension.c_str());
		file.close();
		return false;
	}
	file.close();
	return true;
}

template <typename Scalar>
inline bool writeOBJ(const std::ofstream& obj_file,
	const std::vector<Eigen::Vector3d>& V,
	const std::vector<Eigen::Vector3i>& F)
{
	const size_t vertices = V.size();
	const size_t faces = F.size();
	obj_file << "# Vertices: " << vertices << "\tFaces: " << faces << "\n";
	for (size_t i = 0; i < vertices; i++)
	{
		auto v = V[i];
		obj_file << "v " << std::fixed << std::setprecision(5) << v.x() << " " << v.y() << " " << v.z() << "\n";
	}
	for (size_t i = 0; i < faces; i++)
	{
		auto f = F[i];
		obj_file << "f " << (f + Eigen::Vector3i(1, 1, 1)).transpose() << "\n";
	}
}

template <typename Scalar>
inline bool writeOFF(const std::ofstream& off_file,
	const std::vector<Eigen::Vector3d>& V,
	const std::vector<Eigen::Vector3i>& F)
{
	const size_t vertices = V.size();
	const size_t faces = F.size();
	off_file << "OFF\n" << vertices << " " << faces << " 0\n";
	for (size_t i = 0; i < vertices; i++)
	{
		auto v = V[i];
		off_file << std::fixed << std::setprecision(5) << v.x() << " " << v.y() << " " << v.z() << "\n";
	}
	for (size_t i = 0; i < faces; i++)
	{
		auto f = F[i];
		off_file << "3 " << (f + Eigen::Vector3i(1, 1, 1)).transpose() << "\n";
	}
}

template <typename Scalar>
inline bool writeMeshFile(const string& filePath,
	const std::vector<Eigen::Vector3d>& V,
	const std::vector<Eigen::Vector3i>& F)
{
	std::ofstream obj_file(filePath, std::ios::out);
	if (!obj_file.is_open())
	{
		fprintf(stderr, "IOError: could not open %s\n",
			filePath.c_str());
		return false;
	}
	if (V.cols() != 3)
	{
		fprintf(stderr, "Error: Verts should have 3 columns\n");
		return false;
	}

	string extension = getFileExtension(filePath);
	if (extension == "obj")
	{
		if (!writeOBJ(file, V, F))
		{
			file.close();
			return false;
		}
	}
	else if (extension == "off")
	{
		if (!writeOFF(file, V, F))
		{
			file.close();
			return false;
		}
	}
	else
	{
		fprintf(stderr, "Files with suffix \'%s\' are not supported!\Vn\n", extension.c_str());
		file.close();
		return false;
	}
	file.close();
	return true;
}

template <typename Scalar>
inline bool writeOBJ(const std::ofstream& obj_file,
	const std::vector<std::vector<Scalar>>& V,
	const std::vector<std::vector<int>>& F)
{
	const size_t vertices = V.size();
	const size_t faces = F.size();
	obj_file << "# Vertices: " << vertices << "\tFaces: " << faces << "\n";
	for (size_t i = 0; i < vertices; i++)
	{
		obj_file << "v " << std::fixed << std::setprecision(5) << V[i][0] << " " << V[i][1] << " " << V[i][2] << "\n";
	}
	for (size_t i = 0; i < faces; i++)
	{
		obj_file << "f " << std::fixed << std::setprecision(5) << F[i][0] << " " << F[i][1] << " " << F[i][2] << "\n";
	}
}

template <typename Scalar>
inline bool writeOFF(const std::ofstream& off_file,
	const std::vector<std::vector<Scalar>>& V,
	const std::vector<std::vector<int>>& F)
{
	const size_t vertices = V.size();
	const size_t faces = F.size();
	off_file << "OFF\n" << vertices << " " << faces << " 0\n";
	for (size_t i = 0; i < V.size(); i++)
	{
		off_file << std::fixed << std::setprecision(5) << V[i][0] << " " << V[i][1] << " " << V[i][2] << "\n";
	}
	for (size_t i = 0; i < F.size(); i++)
	{
		off_file << "3 " << std::fixed << std::setprecision(5) << F[i][0] << " " << F[i][1] << " " << F[i][2] << "\n";
	}
}

template <typename Scalar>
inline bool writeMeshFile(const string& filePath,
	const std::vector<std::vector<Scalar>>& V,
	const std::vector<std::vector<int>>& F)
{
	std::ofstream file(filePath, std::ios::out);
	if (!file.is_open())
	{
		fprintf(stderr, "IOError: could not open %s\n",
			filePath.c_str());
		return false;
	}
	if (V.cols() != 3)
	{
		fprintf(stderr, "Error: Verts should have 3 columns\n");
		return false;
	}

	string extension = getFileExtension(filePath);
	if (extension == "obj")
	{
		if (!writeOBJ(file, V, F))
		{
			file.close();
			return false;
		}
	}
	else if (extension == "off")
	{
		if (!writeOFF(file, V, F))
		{
			file.close();
			return false;
		}
	}
	else
	{
		fprintf(stderr, "Files with suffix \'%s\' are not supported!\Vn\n", extension.c_str());
		file.close();
		return false;
	}
	file.close();
	return true;
}