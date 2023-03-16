#pragma once
#include <Eigen\dense>

using Eigen;

struct BoundingBox
{
	Vector3d boxOrigin;
	Vector3d boxEnd;
	Vector3d boxWidth;

	BoundingBox() {}

	BoundingBox(const Vector3d& _origin, const Vector3d& _end) :boxOrigin(_origin), boxEnd(_end) { boxWidth = boxEnd - boxOrigin; }
};

struct Triangle
{
	Vector3d p1, p2, p3;

	Triangle() = default;

	Triangle(const Vector3d& _p1, const Vector3d& _p2, const Vector3d& _p3) :p1(_p1), p2(_p2), p3(_p3) {}

	Vector3d cpCoefficientOfTriangle(const double& x, const double& y, const int& flag)
	{
		Vector3d d1 = p2 - p1;
		Vector3d d2 = p3 - p1;
		if (flag == 0)
		{
			Vector3d intersection = Vector3d(x, y, 0.0);
			double area = d1.cross(d2).z();
			if (abs(area) < 1e-10) return Vector3d(-1.0, -1.0, -1.0);
			Vector3d v1 = p1 - intersection;
			Vector3d v2 = p2 - intersection;
			Vector3d v3 = p3 - intersection;

			double area1 = v2.cross(v3).z();
			double lambda1 = area1 / area;
			if (lambda1 < 0) return Vector3d(-1.0, -1.0, -1.0);
			area1 = v3.cross(v1).z();
			double lambda2 = area1 / area;
			if (lambda2 < 0) return Vector3d(-1.0, -1.0, -1.0);
			area1 = v1.cross(v2).z();
			double lambda3 = area1 / area;
			if (lambda3 < 0) return Vector3d(-1.0, -1.0, -1.0);
			return Vector3d(lambda1, lambda2, lambda3);
		}
		if (flag == 1)
		{
			Vector3d intersection(0.0, x, y);
			double area = d1.cross(d2).x();
			if (abs(area) < 1e-10) return Vector3d(-1.0, -1.0, -1.0);
			Vector3d v1 = p1 - intersection;
			Vector3d v2 = p2 - intersection;
			Vector3d v3 = p3 - intersection;

			double area1 = v2.cross(v3).x();
			double lambda1 = area1 / area;
			if (lambda1 < 0) return Vector3d(-1.0, -1.0, -1.0);
			area1 = v3.cross(v1).x();
			double lambda2 = area1 / area;
			if (lambda2 < 0) return Vector3d(-1.0, -1.0, -1.0);
			area1 = v1.cross(v2).x();
			double lambda3 = area1 / area;
			if (lambda3 < 0) return Vector3d(-1.0, -1.0, -1.0);
			return Vector3d(lambda1, lambda2, lambda3);
		}

		Vector3d intersection = Vector3d(y, 0.0, x);
		double area = d1.cross(d2).y();
		if (abs(area) < 1e-10) return Vector3d(-1.0, -1.0, -1.0);
		Vector3d v1 = p1 - intersection;
		Vector3d v2 = p2 - intersection;
		Vector3d v3 = p3 - intersection;

		double area1 = v2.cross(v3).y();
		double lambda1 = area1 / area;
		if (lambda1 < 0) return Vector3d(-1.0, -1.0, -1.0);
		area1 = v3.cross(v1).y();
		double lambda2 = area1 / area;
		if (lambda2 < 0) return Vector3d(-1.0, -1.0, -1.0);
		area1 = v1.cross(v2).y();
		double lambda3 = area1 / area;
		if (lambda3 < 0) return Vector3d(-1.0, -1.0, -1.0);
		return Vector3d(lambda1, lambda2, lambda3);
	}
};
