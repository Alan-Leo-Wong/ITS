#pragma once

#include "Config.hpp"
#include <fstream>
#include <Eigen/Dense>

NAMESPACE_BEGIN(ITS)

    /**
     * An Axis Aligned Box (AAB) of a certain Real -
     * to be initialized with a boxOrigin and boxEnd
     * @tparam Real
     */
    template<typename Real>
    struct AABox {
        //using Real = typename Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

        Real boxOrigin;
        Real boxEnd;
        Real boxWidth;

        CUDA_GENERAL_CALL AABox() : boxOrigin(Real()), boxEnd(Real()), boxWidth(Real()) {}

        CUDA_GENERAL_CALL AABox(const Real &_boxOrigin, const Real &_boxEnd) : boxOrigin(_boxOrigin), boxEnd(_boxEnd),
                                                                               boxWidth(_boxEnd - _boxOrigin) {}

        CUDA_GENERAL_CALL void scaleAndTranslate(const double &scale_factor, const V3d &translation) {
            const Real center = (boxOrigin + boxEnd) / 2.0;

            const Real scaled_min_point = (boxOrigin - center) * scale_factor + center + translation;
            const Real scaled_max_point = (boxEnd - center) * scale_factor + center + translation;

            boxOrigin = scaled_min_point;
            boxEnd = scaled_max_point;
        }

        CUDA_GENERAL_CALL AABox<Real>(const AABox<Real> &_box) {
            boxOrigin = _box.boxOrigin;
            boxEnd = _box.boxEnd;
            boxWidth = _box.boxWidth;
        }

        CUDA_GENERAL_CALL AABox<Real> &operator=(const AABox<Real> &_box) {
            boxOrigin = _box.boxOrigin;
            boxEnd = _box.boxEnd;
            boxWidth = _box.boxWidth;

            return *this;
        }
    };

    template<typename Real>
    struct Triangle {
        Real p1, p2, p3;
        Real normal;
        double area;
        double dir;

        CUDA_GENERAL_CALL Triangle() {}

        CUDA_GENERAL_CALL Triangle(const Real &_p1, const Real &_p2, const Real &_p3) : p1(_p1), p2(_p2), p3(_p3) {}
    };

NAMESPACE_END(ITS)
