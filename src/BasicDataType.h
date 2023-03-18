#pragma once
#include <map>
#include <set>
#include <limits>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <Eigen\dense>

using UINT = unsigned int;

constexpr double DINF = (std::numeric_limits<double>::max)();

using V3d = Eigen::Vector3d;
using V2i = Eigen::Vector2i;
using V3i = Eigen::Vector3i;
using VXi = Eigen::VectorXi;
using MXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VXd = Eigen::VectorXd;

using PII = std::pair<int, int>;
using PDD = std::pair<double, double>;
using PV3d = std::pair<V3d, V3d>;

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
