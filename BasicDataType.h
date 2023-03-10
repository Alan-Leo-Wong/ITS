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

constexpr double DINF = (std::numeric_limits<double>::max)();

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

struct Triangle
{
	V3d p1, p2, p3;

	Triangle() = default;

	Triangle(const V3d& _p1, const V3d& _p2, const V3d& _p3) :p1(_p1), p2(_p2), p3(_p3) {}

	V3d cpCoefficientOfTriangle(const double& x, const double& y, int flag)
	{
		V3d d1 = p2 - p1;
		V3d d2 = p3 - p1;
		if (flag == 0)
		{
			V3d intersection = V3d(x, y, 0.0);
			double area = d1.cross(d2).z();
			if (abs(area) < 1e-10) return V3d(-1.0, -1.0, -1.0);
			V3d v1 = p1 - intersection;
			V3d v2 = p2 - intersection;
			V3d v3 = p3 - intersection;

			double area1 = v2.cross(v3).z();
			double lambda1 = area1 / area;
			if (lambda1 < 0) return V3d(-1.0, -1.0, -1.0);
			area1 = v3.cross(v1).z();
			double lambda2 = area1 / area;
			if (lambda2 < 0) return V3d(-1.0, -1.0, -1.0);
			area1 = v1.cross(v2).z();
			double lambda3 = area1 / area;
			if (lambda3 < 0) return V3d(-1.0, -1.0, -1.0);
			return V3d(lambda1, lambda2, lambda3);
		}
		if (flag == 1)
		{
			V3d intersection(0.0, x, y);
			double area = d1.cross(d2).x();
			if (abs(area) < 1e-10) return V3d(-1.0, -1.0, -1.0);
			V3d v1 = p1 - intersection;
			V3d v2 = p2 - intersection;
			V3d v3 = p3 - intersection;

			double area1 = v2.cross(v3).x();
			double lambda1 = area1 / area;
			if (lambda1 < 0) return V3d(-1.0, -1.0, -1.0);
			area1 = v3.cross(v1).x();
			double lambda2 = area1 / area;
			if (lambda2 < 0) return V3d(-1.0, -1.0, -1.0);
			area1 = v1.cross(v2).x();
			double lambda3 = area1 / area;
			if (lambda3 < 0) return V3d(-1.0, -1.0, -1.0);
			return V3d(lambda1, lambda2, lambda3);
		}

		V3d intersection = V3d(y, 0.0, x);
		double area = d1.cross(d2).y();
		if (abs(area) < 1e-10) return V3d(-1.0, -1.0, -1.0);
		V3d v1 = p1 - intersection;
		V3d v2 = p2 - intersection;
		V3d v3 = p3 - intersection;

		double area1 = v2.cross(v3).y();
		double lambda1 = area1 / area;
		if (lambda1 < 0) return V3d(-1.0, -1.0, -1.0);
		area1 = v3.cross(v1).y();
		double lambda2 = area1 / area;
		if (lambda2 < 0) return V3d(-1.0, -1.0, -1.0);
		area1 = v1.cross(v2).y();
		double lambda3 = area1 / area;
		if (lambda3 < 0) return V3d(-1.0, -1.0, -1.0);
		return V3d(lambda1, lambda2, lambda3);
	}
};