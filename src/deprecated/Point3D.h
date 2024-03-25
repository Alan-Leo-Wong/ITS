#pragma once
#include "BasicDataType.h"
#include "cuAcc/CUDAMacro.h"

struct Point
{
public:
	V3d coords;

public:
	// constructor
	Point() {}
	CUDA_CALLABLE_MEMBER Point(const V3d& _coords) : coords(_coords) {}
	CUDA_CALLABLE_MEMBER Point(const double& x, const double& y, const double& z) : coords(V3d(x, y, z)) {}

	// unary op
	CUDA_CALLABLE_MEMBER inline       double& operator[] (unsigned int i) { return coords[i]; }
	CUDA_CALLABLE_MEMBER inline const double& operator[] (unsigned int i) const { return coords[i]; }
	CUDA_CALLABLE_MEMBER inline       double& operator() (unsigned int i) { return coords[i]; }
	CUDA_CALLABLE_MEMBER inline const double& operator() (unsigned int i) const { return coords[i]; }
	CUDA_CALLABLE_MEMBER inline Point operator - () const { Point q; q.coords = -coords; return q; }
	CUDA_CALLABLE_MEMBER inline const double& x() { return coords.x(); }
	CUDA_CALLABLE_MEMBER inline const double& y() { return coords.y(); }
	CUDA_CALLABLE_MEMBER inline const double& z() { return coords.z(); }

	// binary op
	CUDA_CALLABLE_MEMBER inline Point  operator+ (const Point& other) const { Point p; p.coords = coords + other.coords; return p; }
	CUDA_CALLABLE_MEMBER inline Point& operator+=(const Point& other) { return Point(other.coords + coords); }
	CUDA_CALLABLE_MEMBER inline Point  operator- (const Point& other) const { return (*this).coords + (-other.coords); }
	CUDA_CALLABLE_MEMBER inline Point& operator-=(const Point& other) { return Point(other.coords + coords); }



	friend std::ostream& operator << (std::ostream& os, const Point& p);
};

std::ostream& operator << (std::ostream& os, const Point& p)
{
	os << "x = " << p.coords.x() << ", y = " << p.coords.y() << ", z = " << p.coords.z() << endl;
	return os;
}