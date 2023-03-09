#pragma once
#include <iostream>
#include <Eigen\dense>
#include <vector>
#include <string>
#include <fstream>
#include <map>
#include <set>

using V3d = Eigen::Vector3d;
using V2i = Eigen::Vector2i;
using V3i = Eigen::Vector3i;
using VXi = Eigen::VectorXi;
using MXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using SpMat = Eigen::SparseMatrix<double>;
using VXd = Eigen::VectorXd;
using std::vector;
using std::string;
using std::map;
using std::set;
using std::make_pair;
using std::min;
using std::max;
using std::ifstream;
using std::cout;
using std::endl;
using PII = std::pair<int, int>;
using PDD = std::pair<double, double>;
using PV3d = std::pair<V3d, V3d>;

//struct double3
//{
//	double x, y, z;
//
//	double3() = default;
//
//	double3(const double& _x, const double& _y, const double& _z) :x(_x), y(_y), z(_z) {}
//
//	double3& operator=(const double3& other) { x = other.x, y = other.y, z = other.z; }
//};