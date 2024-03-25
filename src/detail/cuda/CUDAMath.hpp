#pragma once

#include "Config.hpp"
#include "cuda_runtime.h"
#include <cmath>
#include <array>
#include <Eigen/Core>

NAMESPACE_BEGIN(ITS)

    using uint = unsigned int;

    inline bool isPow2(const unsigned int &x) { return ((x & (x - 1)) == 0); }

    inline unsigned int nextPow2(unsigned int x) {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }

    inline __host__ __device__ bool operator==(const uint3 &a, const uint3 &b) {
        return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
    }

    inline __host__ __device__ bool operator!=(const uint3 &a, const uint3 &b) {
        return (!(a == b));
    }

    // constructor
    inline __host__ __device__ uint3 make_uint3(const Eigen::VectorXi &a) {
        return ::make_uint3(uint(a.x()), uint(a.y()), uint(a.z()));
    }

    inline __host__ __device__ double3 make_double3(const Eigen::VectorXd &a) {
        return ::make_double3(a.x(), a.y(), a.z());
    }

    // addition
    inline __host__ __device__ float3 operator+(const float3 &a, const float3 &b) {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    inline __host__ __device__ void operator+=(float3 &a, const float3 &b) {
        a.x += b.x;
        a.y += b.y;
        a.z += b.z;
    }

    inline __host__ __device__ float3 operator+(const float3 &a, const float &b) {
        return make_float3(a.x + b, a.y + b, a.z + b);
    }

    inline __host__ __device__ void operator+=(float3 &a, const float &b) {
        a.x += b;
        a.y += b;
        a.z += b;
    }

    inline __host__ __device__ double3 operator+(const double3 &a,
                                                 const double3 &b) {
        return ::make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    inline __host__ __device__ void operator+=(double3 &a, const double3 &b) {
        a.x += b.x;
        a.y += b.y;
        a.z += b.z;
    }

    inline __host__ __device__ double3 operator+(const double3 &a, double &b) {
        return ::make_double3(a.x + b, a.y + b, a.z + b);
    }

    inline __host__ __device__ void operator+=(double3 &a, const double &b) {
        a.x += b;
        a.y += b;
        a.z += b;
    }

    inline __host__ __device__ int3 operator+(const int3 a, const int3 &b) {
        return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    inline __host__ __device__ void operator+=(int3 &a, const int3 &b) {
        a.x += b.x;
        a.y += b.y;
        a.z += b.z;
    }

    inline __host__ __device__ int3 operator+(const int3 a, const int &b) {
        return make_int3(a.x + b, a.y + b, a.z + b);
    }

    inline __host__ __device__ void operator+=(int3 &a, const int &b) {
        a.x += b;
        a.y += b;
        a.z += b;
    }

    inline __host__ __device__ uint3 operator+(const uint3 a, const uint3 &b) {
        return ::make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    inline __host__ __device__ void operator+=(uint3 &a, const uint3 &b) {
        a.x += b.x;
        a.y += b.y;
        a.z += b.z;
    }

    inline __host__ __device__ uint3 operator+(const uint3 a, const uint &b) {
        return ::make_uint3(a.x + b, a.y + b, a.z + b);
    }

    inline __host__ __device__ void operator+=(uint3 &a, const uint &b) {
        a.x += b;
        a.y += b;
        a.z += b;
    }

    inline CUDA_GENERAL_CALL float fminf(float a, float b) {
        return a < b ? a : b;
    }

    inline CUDA_GENERAL_CALL float fmaxf(float a, float b) {
        return a > b ? a : b;
    }

    inline CUDA_GENERAL_CALL double fminf(double a, double b) {
        return a < b ? a : b;
    }

    inline CUDA_GENERAL_CALL double fmaxf(double a, double b) {
        return a > b ? a : b;
    }

    inline CUDA_GENERAL_CALL int mini(int a, int b) {
        return a < b ? a : b;
    }

    inline CUDA_GENERAL_CALL int maxi(int a, int b) {
        return a > b ? a : b;
    }

    inline CUDA_GENERAL_CALL Eigen::Vector3f fminf(Eigen::Vector3f a, Eigen::Vector3f b) {
        return Eigen::Vector3f(fminf(a.x(), b.x()), fminf(a.y(), b.y()), fminf(a.z(), b.z()));
    }

    inline CUDA_GENERAL_CALL Eigen::Vector3f fmaxf(Eigen::Vector3f a, Eigen::Vector3f b) {
        return Eigen::Vector3f(fmaxf(a.x(), b.x()), fmaxf(a.y(), b.y()), fmaxf(a.z(), b.z()));
    }

    inline CUDA_GENERAL_CALL Eigen::Vector3d fminf(Eigen::Vector3d a, Eigen::Vector3d b) {
        return Eigen::Vector3d(fminf(a.x(), b.x()), fminf(a.y(), b.y()), fminf(a.z(), b.z()));
    }

    inline CUDA_GENERAL_CALL Eigen::Vector3d fmaxf(Eigen::Vector3d a, Eigen::Vector3d b) {
        return Eigen::Vector3d(fmaxf(a.x(), b.x()), fmaxf(a.y(), b.y()), fmaxf(a.z(), b.z()));
    }

    inline CUDA_GENERAL_CALL Eigen::Vector3i vmini(Eigen::Vector3i a, Eigen::Vector3i b) {
        return Eigen::Vector3i(mini(a.x(), b.x()), mini(a.y(), b.y()), mini(a.z(), b.z()));
    }

    inline CUDA_GENERAL_CALL Eigen::Vector3i vmaxi(Eigen::Vector3i a, Eigen::Vector3i b) {
        return Eigen::Vector3i(maxi(a.x(), b.x()), maxi(a.y(), b.y()), maxi(a.z(), b.z()));
    }

    inline CUDA_GENERAL_CALL float clamp(float f, float a, float b) {
        return fmaxf(a, fminf(f, b));
    }

    inline CUDA_GENERAL_CALL double clamp(double f, double a, double b) {
        return fmaxf(a, fminf(f, b));
    }

    inline CUDA_GENERAL_CALL int clamp(int f, int a, int b) {
        return maxi(a, mini(f, b));
    }

    inline CUDA_GENERAL_CALL Eigen::Vector3f clamp(Eigen::Vector3f v, Eigen::Vector3f a, Eigen::Vector3f b) {
        return Eigen::Vector3f(clamp(v.x(), a.x(), b.x()), clamp(v.y(), a.y(), b.y()), clamp(v.z(), a.z(), b.z()));
    }

    inline CUDA_GENERAL_CALL Eigen::Vector3d clamp(Eigen::Vector3d v, Eigen::Vector3d a, Eigen::Vector3d b) {
        return Eigen::Vector3d(clamp(v.x(), a.x(), b.x()), clamp(v.y(), a.y(), b.y()), clamp(v.z(), a.z(), b.z()));
    }

    inline CUDA_GENERAL_CALL Eigen::Vector3i clamp(Eigen::Vector3i v, Eigen::Vector3i a, Eigen::Vector3i b) {
        return Eigen::Vector3i(clamp(v.x(), a.x(), b.x()), clamp(v.y(), a.y(), b.y()), clamp(v.z(), a.z(), b.z()));
    }

    template<typename T>
    CUDA_GENERAL_CALL T _lerp(const T &x0, const T &x1, const double &t) {
        return x0 + t * (x1 - x0);
    }

    template<typename T>
    CUDA_GENERAL_CALL T bi_lerp(const T &x0, const T &x1,
                                const T &y0, const T &y1,
                                const double &tx, const double &ty) {
        // x0------x1
        //  |      |
        //  |  	   |
        // y0------y1
        //
        // auto nx0 = _lerp(x0, x1, tx);
        // auto nx1 = _lerp(y0, y1, tx);
        // return _lerp(nx0, nx1, ty);
        return x0 * (1 - tx) * (1 - ty) + x1 * tx * (1 - ty) + y0 * (1 - tx) * ty + y1 * tx * ty;
    }

    template<typename T>
    CUDA_GENERAL_CALL T tri_lerp(const std::array<double, 8> &val,
                                 const double &tx, const double &ty, const double &tz) {
        //	   4----a----6
        //	  /|        /|
        //	 / |       / |
        //	5--|-c----7  |
        //	|  0----b-|--2
        //	| /       | /
        //	1----d----3

        auto a = _lerp(val[4], val[6], ty);
        auto b = _lerp(val[0], val[2], ty);
        auto c = _lerp(val[5], val[7], ty);
        auto d = _lerp(val[1], val[3], ty);
        return bi_lerp(a, b, c, d, tz, tx);
    }

NAMESPACE_END(ITS)
