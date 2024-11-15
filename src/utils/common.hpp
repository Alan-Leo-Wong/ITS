#pragma once
#include "../Config.h"
#include "Geometry.hpp"
#include "cuda/CUDAMacro.h"
#include <string>
#include <random>
#include <iostream>

namespace tType {
	const std::string NONE("0"), BOLD("1"), DIM("2"), UNDERLINE("4"), BLINK("5"),
		INVERSE("7"), HIDDEN("8");
}
namespace tColor {
	const std::string BLACK("30"), RED("31"), GREEN("32"), YELLOW("33"), BLUE("34"),
		MAGENTA("35"), CYAN("36");
}

//namespace qp {
//	template <typename T, typename... Ts> void qp_ctrl(T v, Ts... vl) {
//		std::string ctrl("\033[");
//		ctrl += std::string(v);
//		if constexpr (sizeof...(vl) > 0) {
//			std::array cl = { std::string(vl)... };
//			for (auto& c : cl)
//				ctrl += ";" + c;
//		}
//		ctrl += "m";
//		std::cout << ctrl;
//	}
//	void qp_ctrl() { std::cout << "\033[0m"; }
//
//	// Print the values no line break.
//	template <typename T, typename... Ts> void qprint_nlb(T v, Ts... vl) {
//		std::cout << v << " ";
//		if constexpr (sizeof...(vl) > 0) {
//			qprint_nlb(vl...);
//			return;
//		}
//		// std::cout << std::endl;
//	}
//
//	// Print the values with line break.
//	template <typename T, typename... Ts> void qprint(T v, Ts... vl) {
//		std::cout << v << " ";
//		if constexpr (sizeof...(vl) > 0) {
//			qprint(vl...);
//			return;
//		}
//		std::cout << std::endl;
//	}
//
//	inline void qprint() { printf("\n"); }
//}

namespace detail {
	template <class T, T... inds, class F>
	constexpr _CUDA_GENERAL_CALL_ _FORCE_INLINE_ void
		loop(std::integer_sequence<T, inds...>, F&& f) {
		(f(std::integral_constant<T, inds>{}), ...);
	}

	template <class T, T count, class F>
	constexpr _CUDA_GENERAL_CALL_ _FORCE_INLINE_ void Loop(F&& f) {
		loop(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
	}
} // namespace detail

// used to check colums
template<typename T>
inline int min_size(const std::vector<T>& V)
{
	int min_size = -1;
	for (
		typename std::vector<T>::const_iterator iter = V.begin();
		iter != V.end();
		iter++)
	{
		int size = (int)iter->size();
		// have to handle base case
		if (min_size == -1)
		{
			min_size = size;
		}
		else
		{
			min_size = (min_size < size ? min_size : size);
		}
	}
	return min_size;
}

template <typename T>
inline int max_size(const std::vector<T>& V)
{
	int max_size = -1;
	for (
		typename std::vector<T>::const_iterator iter = V.begin();
		iter != V.end();
		iter++)
	{
		int size = (int)iter->size();
		max_size = (max_size > size ? max_size : size);
	}
	return max_size;
}

template <typename Scalar, int Size>
inline int min_size(const std::vector<Eigen::Matrix<Scalar, Size, 1>>& V)
{
	int min_size = -1;
	for (
		typename std::vector<Eigen::Matrix<Scalar, Size, 1>>::const_iterator iter = V.begin();
		iter != V.end();
		iter++)
	{
		int size = (int)iter->rows();
		// have to handle base case
		if (min_size == -1)
		{
			min_size = size;
		}
		else
		{
			min_size = (min_size < size ? min_size : size);
		}
	}
	return min_size;
}

template <typename Scalar, int Size>
inline int max_size(const std::vector<Eigen::Matrix<Scalar, Size, 1>>& V)
{
	int max_size = -1;
	for (
		typename std::vector<Eigen::Matrix<Scalar, Size, 1>>::const_iterator iter = V.begin();
		iter != V.end();
		iter++)
	{
		int size = (int)iter->rows();
		max_size = (max_size > size ? max_size : size);
	}
	return max_size;
}

template <typename T>
_CUDA_GENERAL_CALL_
inline bool isInRange(const double& l, const double& r, const T& query) {
	return l <= query && query <= r;
}

template <typename... T>
_CUDA_GENERAL_CALL_
inline bool isInRange(const double& l, const double& r, const T &...query) {
	return isInRange(query...);
}

_CUDA_GENERAL_CALL_
inline bool isEqualDouble(const double& l, const double& r, const double& eps) {
	return fabs(l - r) < eps;
}

_CUDA_GENERAL_CALL_
inline bool isLessDouble(const double& l, const double& r, const double& eps) {
	return (!isEqualDouble(l, r, eps) && l < r);
}

_CUDA_GENERAL_CALL_
inline bool isLargeDouble(const double& l, const double& r, const double& eps) {
	return (!isEqualDouble(l, r, eps) && l > r);
}

template <typename T, typename Derived>
inline bool list2matrix(const std::vector<std::vector<T>>& V, Eigen::PlainObjectBase<Derived>& M)
{
	// number of rows
	int m = V.size();
	if (m == 0)
	{
		M.resize(
			Derived::RowsAtCompileTime >= 0 ? Derived::RowsAtCompileTime : 0,
			Derived::ColsAtCompileTime >= 0 ? Derived::ColsAtCompileTime : 0
		);
		return true;
	}
	// number of columns
	int n = min_size(V);
	if (n != max_size(V))
	{
		return false;
	}
	assert(n != -1);
	// Resize output
	M.resize(m, n);

	// Loop over rows
	for (int i = 0; i < m; i++)
	{
		// Loop over cols
		for (int j = 0; j < n; j++)
		{
			M(i, j) = V[i][j];
		}
	}
	return true;
}

template <typename Derived, typename Scalar, int Size>
inline bool list2Matrix(const std::vector<Eigen::Matrix<Scalar, Size, 1>>& V, Eigen::PlainObjectBase<Derived>& M)
{
	// number of rows
	int m = V.size();
	if (m == 0)
	{
		//fprintf(stderr,"Error: list_to_matrix() list is empty()\n");
		//return false;
		M.resize(
			Derived::RowsAtCompileTime >= 0 ? Derived::RowsAtCompileTime : 0,
			Derived::ColsAtCompileTime >= 0 ? Derived::ColsAtCompileTime : 0
		);
		return true;
	}
	// number of columns
	int n = min_size(V);
	if (n != max_size(V)) return false;
	assert(n != -1);

	// Resize output
	M.resize(m, n);

	// Loop over rows
	for (int i = 0; i < m; i++)
		M.row(i) = V[i];
	return true;
}

template <typename Scalar>
inline bool getGaussianRandomMatrix(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& area,
	const size_t& num, const float& scaleFactor, const float& dis,
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& M)
{
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> _M(num, 3);
	std::default_random_engine e(time(0)); // current time as seed
	std::normal_distribution<Scalar> n(0, 1);
	_M = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(num, 3)
		.unaryExpr([&](Scalar val) { return static_cast<Scalar>(round((n(e) + 1e-6) * 1e6) / 1e6); });

	M.resize(num, 4);
	M.setOnes();
	M.block(0, 0, num, 3) = _M.block(0, 0, num, 3);

	const auto min_area = area.colwise().minCoeff();
	const auto max_area = area.colwise().maxCoeff();
	const auto min_m = M.colwise().minCoeff();
	const auto max_m = M.colwise().maxCoeff();

	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> diag_area = max_area - min_area; // bounding box
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> diag_m = max_m - min_m; // random points

	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> center_area = 0.5 * (max_area + min_area);
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> center_m = 0.5 * (max_m + min_m);

	Eigen::Matrix<Scalar, 4, 4> zoomMatrix = Eigen::Matrix<Scalar, 4, 4>::Identity();
	Eigen::Matrix<Scalar, 4, 4> transMatrix = Eigen::Matrix<Scalar, 4, 4>::Identity();

	for (int d = 0; d < 3; d++) {
		if (diag_m[d] == 0) return false;
		zoomMatrix(d, d) = (diag_area[d] / diag_m[d]) * scaleFactor;
		transMatrix(3, d) = (center_area[d] - center_m[d] * zoomMatrix(d, d)) + dis;
	}

	M = M * zoomMatrix * transMatrix;
	M = M.block(0, 0, num, 3);
	M = (M.array() < 1e-6).select(0, M);

	return true;
}

template <typename Scalar>
inline bool getUniformRandomMatrix(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& area,
	const size_t& num, const float& scaleFactor, const float& dis,
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& M)
{
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> _M(num, 3);
	std::default_random_engine e(time(0)); // current time as seed
	std::uniform_real_distribution<Scalar> n(-1, 1);
	_M = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(num, 3)
		.unaryExpr([&](Scalar val) { return static_cast<Scalar>(round((n(e) + 1e-6) * 1e6) / 1e6); });

	M.resize(num, 4);
	M.setOnes();
	M.block(0, 0, num, 3) = _M.block(0, 0, num, 3);

	const auto min_area = area.colwise().minCoeff();
	const auto max_area = area.colwise().maxCoeff();
	const auto min_m = M.colwise().minCoeff();
	const auto max_m = M.colwise().maxCoeff();

	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> diag_area = max_area - min_area; // bounding box
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> diag_m = max_m - min_m; // random points

	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> center_area = 0.5 * (max_area + min_area);
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> center_m = 0.5 * (max_m + min_m);

	Eigen::Matrix<Scalar, 4, 4> zoomMatrix = Eigen::Matrix<Scalar, 4, 4>::Identity();
	Eigen::Matrix<Scalar, 4, 4> transMatrix = Eigen::Matrix<Scalar, 4, 4>::Identity();

	for (int d = 0; d < 3; d++) {
		if (diag_m[d] == 0) return false;
		zoomMatrix(d, d) = (diag_area[d] / diag_m[d]) * scaleFactor;
		transMatrix(3, d) = (center_area[d] - center_m[d] * zoomMatrix(d, d)) + dis;
	}

	M = M * zoomMatrix * transMatrix;
	M = M.block(0, 0, num, 3);
	M = (M.array() < 1e-6).select(0, M);

	return true;
}

template <typename Scalar>
inline bool getUniformRandomMatrix(const AABox<Scalar>& _area,
	const size_t& num, const double& scaleFactor, const V3d& dis, vector<Scalar>& randomPoints)
{
	if (scaleFactor <= .0) return false;

	//using BoxType = AABox<Scalar>::type;

	AABox<Scalar> area = _area;
	area.scaleAndTranslate(scaleFactor, dis);

	// 生成Halton序列的第index个值
	auto haltonSequence = [](int index, int base)
	{
		double result = 0.0;
		double f = 1.0 / base;
		int i = index;

		while (i > 0)
		{
			result += f * (i % base);
			i = std::floor(i / base);
			f /= base;
		}

		return result;
	};

	// 将Halton序列值映射到[min, max]范围内
	auto mapToRange = [](double value, double min, double max)
	{
		return min + value * (max - min);
	};

	// 在[minArea, maxArea]范围内进行蓝噪声采样
	const Scalar& minArea = area.boxOrigin;
	const Scalar& maxArea = area.boxEnd;

	int baseX = 2; // X轴上的基数
	int baseY = 3; // Y轴上的基数
	int baseZ = 5; // Z轴上的基数

	for (int i = 0; i < num; ++i)
	{
		double x = mapToRange(haltonSequence(i, baseX), minArea.x(), maxArea.x());
		double y = mapToRange(haltonSequence(i, baseY), minArea.y(), maxArea.y());
		double z = mapToRange(haltonSequence(i, baseZ), minArea.z(), maxArea.z());
		randomPoints.emplace_back(Scalar(x, y, z));
	}

	return true;
}

template <typename Scalar>
inline bool getUniformRandomMatrix(const AABox<Scalar>& _area,
	const size_t& num, const float& scaleFactor, const V3f& dis, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& M)
{
	vector<Scalar> randomPoints;
	if (getUniformRandomMatrix<Scalar>(_area, num, scaleFactor, dis, randomPoints))
	{
		Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 3, Eigen::RowMajor>> mat(reinterpret_cast<double*>(randomPoints.data()), num, 3);
		M = mat;
		return true;
	}
	return false;
}

template <typename Scalar>
inline bool getGaussianRandomMatrix(const AABox<Scalar>& _area,
	const size_t& num, const double& scaleFactor, const V3d& dis, vector<Scalar>& randomPoints)
{
	if (scaleFactor <= .0) return false;

	AABox<Scalar> area = _area;
	area.scaleAndTranslate(scaleFactor, dis);

	static std::random_device rd; // 创建一个真随机数生成器
	// 伪随机数生成器gen (不适用rd生成随机数是因为std::random_device可能会产生比较慢的真随机数)
	//static std::mt19937 gen(rd()); // 使用真随机数生成器生成种子(通过rd())，将该种子传递给std::mt19937对象gen进行初始化
	static std::mt19937 gen(1314); // 使用常数种子传递给std::mt19937对象gen进行初始化
	auto gaussianSample = [](const Scalar& mean, const Scalar& stddev) {
		std::normal_distribution<double> dist(0.0, 1.0);

		Scalar sample;
		for (int i = 0; i < 3; ++i)
			sample(i) = mean(i) + stddev(i) * dist(gen);

		return sample;
	};

	const Scalar& minArea = area.boxOrigin;
	const Scalar& maxArea = area.boxEnd;

	Scalar mean = (maxArea + minArea) / 2.0;
	Scalar stddev = (maxArea - minArea) / 6.0;

	for (int i = 0; i < num; ++i)
	{
		// 生成高斯样本
		Scalar sample = gaussianSample(mean, stddev);
		randomPoints.emplace_back(sample);
	}

	return true;
}

template <typename Scalar>
inline bool getGaussianRandomMatrix(const Eigen::Matrix<Scalar, 3, 1>& min_area,
	const Eigen::Matrix<Scalar, 3, 1>& max_area, const size_t& num,
	const float& scaleFactor, const float& dis,
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& M)
{
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> _M(num, 3);
	std::default_random_engine e(1314); // current time as seed
	std::normal_distribution<Scalar> n(-1, 1);
	_M = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(num, 3)
		.unaryExpr([&](Scalar val) { return static_cast<Scalar>(round((n(e) + 1e-6) * 1e6) / 1e6); });

	_M.conservativeResize(num + 1, 4); // strange value because reallocate memory
	Eigen::Matrix<Scalar, 1, 4> r_zeroVec(num);
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> c_onesVec(num + 1);
	r_zeroVec.setZero(); c_onesVec.setOnes();
	_M.row(num) = r_zeroVec; _M.col(3) = c_onesVec;

	const auto min_m = _M.colwise().minCoeff();
	const auto max_m = _M.colwise().maxCoeff();

	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> diag_area = max_area - min_area; // bounding box
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> diag_m = max_m - min_m; // random points

	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> center_area = 0.5 * (max_area + min_area);
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> center_m = 0.5 * (max_m + min_m);

	Eigen::Matrix<Scalar, 4, 4> zoomMatrix = Eigen::Matrix<Scalar, 4, 4>::Identity();
	Eigen::Matrix<Scalar, 4, 4> transMatrix = Eigen::Matrix<Scalar, 4, 4>::Identity();

	for (int d = 0; d < 3; d++)
	{
		if (diag_m[d] == 0) return false;
		zoomMatrix(d, d) = (diag_area[d] / diag_m[d]) * scaleFactor;
		transMatrix(3, d) = (center_area[d] - center_m[d] * zoomMatrix(d, d)) + dis;
	}

	_M = _M * zoomMatrix * transMatrix;
	M.resize(num, 3);
	M = _M.block(0, 0, num, 3);
	M = (M.array().abs() < 1e-9).select(0, M);

	return true;
}

template <typename Scalar>
inline bool getUniformRandomMatrix(const Eigen::Matrix<Scalar, 3, 1>& min_area,
	const Eigen::Matrix<Scalar, 3, 1>& max_area, const size_t& num,
	const float& scaleFactor, const float& dis,
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& M)
{
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> _M(num, 3);
	//std::default_random_engine e(time(0)); // current time as seed
	std::default_random_engine e(1314);
	std::uniform_real_distribution<Scalar> n(-1, 1);
	_M = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(num, 3)
		.unaryExpr([&](Scalar val) { return static_cast<Scalar>(round((n(e) + 1e-6) * 1e6) / 1e6); });

	_M.conservativeResize(num + 1, 4); // strange value because reallocate memory
	Eigen::Matrix<Scalar, 1, 4> r_zeroVec(num);
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> c_onesVec(num + 1);
	r_zeroVec.setZero(); c_onesVec.setOnes();
	_M.row(num) = r_zeroVec; _M.col(3) = c_onesVec;

	const auto min_m = _M.colwise().minCoeff();
	const auto max_m = _M.colwise().maxCoeff();

	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> diag_area = max_area - min_area; // bounding box
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> diag_m = max_m - min_m; // random points

	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> center_area = 0.5 * (max_area + min_area);
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> center_m = 0.5 * (max_m + min_m);

	Eigen::Matrix<Scalar, 4, 4> zoomMatrix = Eigen::Matrix<Scalar, 4, 4>::Identity();
	Eigen::Matrix<Scalar, 4, 4> transMatrix = Eigen::Matrix<Scalar, 4, 4>::Identity();

	for (int d = 0; d < 3; d++)
	{
		if (diag_m[d] == 0) return false;
		zoomMatrix(d, d) = (diag_area[d] / diag_m[d]) * scaleFactor;
		transMatrix(3, d) = (center_area[d] - center_m[d] * zoomMatrix(d, d)) + dis;
	}

	_M = _M * zoomMatrix * transMatrix;
	M.resize(num, 3);
	M = _M.block(0, 0, num, 3);
	M = (M.array().abs() < 1e-9).select(0, M);

	return true;
}