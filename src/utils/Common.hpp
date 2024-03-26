#pragma once

#include "Config.hpp"
#include "detail/Geometry.hpp"
#include <string>
#include <random>
#include <iostream>

NAMESPACE_BEGIN(ITS)
    namespace utils {
        namespace tType {
            const std::string NONE("0"), BOLD("1"), DIM("2"), UNDERLINE("4"), BLINK("5"),
                    INVERSE("7"), HIDDEN("8");
        }
        namespace tColor {
            const std::string BLACK("30"), RED("31"), GREEN("32"), YELLOW("33"), BLUE("34"),
                    MAGENTA("35"), CYAN("36");
        }

        template<class T, T... inds, class F>
        constexpr CUDA_GENERAL_CALL FORCE_INLINE void
        loop(std::integer_sequence<T, inds...>, F &&f) {
            (f(std::integral_constant<T, inds>{}), ...);
        }

        template<class T, T count, class F>
        constexpr CUDA_GENERAL_CALL FORCE_INLINE void Loop(F &&f) {
            loop(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
        }

        // used to check colums
        template<typename T>
        inline int min_size(const std::vector<T> &V) {
            int min_size = -1;
            for (
                    typename std::vector<T>::const_iterator iter = V.begin();
                    iter != V.end();
                    iter++) {
                int size = (int) iter->size();
                // have to handle base case
                if (min_size == -1) {
                    min_size = size;
                } else {
                    min_size = (min_size<size ? min_size : size);
                }
            }
            return min_size;
        }

        template<typename T>
        inline int max_size(const std::vector<T> &V) {
            int max_size = -1;
            for (
                    typename std::vector<T>::const_iterator iter = V.begin();
                    iter != V.end();
                    iter++) {
                int size = (int) iter->size();
                max_size = (max_size > size ? max_size : size);
            }
            return max_size;
        }

        template<typename Scalar, int Size>
        inline int min_size(const std::vector<Eigen::Matrix<Scalar, Size, 1>> &V) {
            int min_size = -1;
            for (
                    typename std::vector<Eigen::Matrix<Scalar, Size, 1>>::const_iterator iter = V.begin();
                    iter != V.end();
                    iter++) {
                int size = (int) iter->rows();
                // have to handle base case
                if (min_size == -1) {
                    min_size = size;
                } else {
                    min_size = (min_size<size ? min_size : size);
                }
            }
            return min_size;
        }

        template<typename Scalar, int Size>
        inline int max_size(const std::vector<Eigen::Matrix<Scalar, Size, 1>> &V) {
            int max_size = -1;
            for (
                    typename std::vector<Eigen::Matrix<Scalar, Size, 1>>::const_iterator iter = V.begin();
                    iter != V.end();
                    iter++) {
                int size = (int) iter->rows();
                max_size = (max_size > size ? max_size : size);
            }
            return max_size;
        }

        template<typename T>
        CUDA_GENERAL_CALL
        inline bool isInRange(const double &l, const double &r, const T &query) {
            return l <= query && query <= r;
        }

        template<typename... T>
        CUDA_GENERAL_CALL
        inline bool isInRange(const double &l, const double &r, const T &...query) {
            return isInRange(query...);
        }

        CUDA_GENERAL_CALL
        inline bool isEqualDouble(const double &l, const double &r, const double &eps) {
            return fabs(l - r) < eps;
        }

        CUDA_GENERAL_CALL
        inline bool isLessDouble(const double &l, const double &r, const double &eps) {
            return (!isEqualDouble(l, r, eps) && l < r);
        }

        CUDA_GENERAL_CALL
        inline bool isLargeDouble(const double &l, const double &r, const double &eps) {
            return (!isEqualDouble(l, r, eps) && l > r);
        }

        template<typename T, typename Derived>
        inline bool list2matrix(const std::vector<std::vector<T>> &V, Eigen::PlainObjectBase<Derived> &M) {
            // number of rows
            int m = V.size();
            if (m == 0) {
                M.resize(
                        Derived::RowsAtCompileTime >= 0 ? Derived::RowsAtCompileTime : 0,
                        Derived::ColsAtCompileTime >= 0 ? Derived::ColsAtCompileTime : 0
                );
                return true;
            }
            // number of columns
            int n = min_size(V);
            if (n != max_size(V)) {
                return false;
            }
            assert(n != -1);
            // Resize output
            M.resize(m, n);

            // Loop over rows
            for (int i = 0; i < m; i++) {
                // Loop over cols
                for (int j = 0; j < n; j++) {
                    M(i, j) = V[i][j];
                }
            }
            return true;
        }

        template<typename Derived, typename Scalar, int Size>
        inline bool
        list2Matrix(const std::vector<Eigen::Matrix<Scalar, Size, 1>> &V, Eigen::PlainObjectBase<Derived> &M) {
            // number of rows
            int m = V.size();
            if (m == 0) {
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

    } // namespace utils
NAMESPACE_END(ITS)