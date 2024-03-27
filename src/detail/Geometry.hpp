#pragma once

#include "Config.hpp"
#include <random>
#include <fstream>
#include <Eigen/Dense>

NAMESPACE_BEGIN(ITS)

    /**
     * Axis Aligned Bounding-Box (AABB)
     * @tparam Scalar
     * @tparam DIM
     * @tparam T
     */
    template<typename Scalar, int DIM, typename T=Eigen::Matrix<Scalar, DIM, 1>>
    struct AABBox {
        using type = T;

        T boxOrigin;
        T boxEnd;
        T boxWidth;

        CUDA_GENERAL_CALL AABBox() : boxOrigin(T()), boxEnd(T()), boxWidth(T()) {}

        CUDA_GENERAL_CALL AABBox(const T &_boxOrigin, const T &_boxEnd) : boxOrigin(_boxOrigin), boxEnd(_boxEnd),
                                                                          boxWidth(_boxEnd - _boxOrigin) {}

        CUDA_GENERAL_CALL void scaleAndTranslate(double scale_factor, const T &translation) {
            T center = (boxOrigin + boxEnd) / 2.0;

            T scaled_min_point = (boxOrigin - center) * scale_factor + center + translation;
            T scaled_max_point = (boxEnd - center) * scale_factor + center + translation;

            boxOrigin = scaled_min_point;
            boxEnd = scaled_max_point;
        }

        CUDA_GENERAL_CALL AABBox<Scalar, DIM, T>(const AABBox<Scalar, DIM, T> &_box) {
            boxOrigin = _box.boxOrigin;
            boxEnd = _box.boxEnd;
            boxWidth = _box.boxWidth;
        }

        CUDA_GENERAL_CALL AABBox<Scalar, DIM, T> &operator=(const AABBox<Scalar, DIM, T> &_box) {
            boxOrigin = _box.boxOrigin;
            boxEnd = _box.boxEnd;
            boxWidth = _box.boxWidth;

            return *this;
        }
    };

    /**
     * 3D Triangle
     * @tparam Scalar
     * @tparam T
     */
    /*template<typename Scalar, typename T=Eigen::Matrix<Scalar, 3, 1>>
    struct Triangle {
        using type = Eigen::Matrix<Scalar, 3, 1>;

        T p1, p2, p3;
        T normal;
        Scalar area;
        Scalar dir;

        CUDA_GENERAL_CALL Triangle() = default;

        CUDA_GENERAL_CALL Triangle(const T &_p1, const T &_p2, const T &_p3) : p1(_p1), p2(_p2), p3(_p3) {}
    };*/
    template<typename T=Eigen::Vector3d>
    struct Triangle {
        using type = T;

        T p1, p2, p3;
        T normal;
        double area;
        double dir;

        CUDA_GENERAL_CALL Triangle() = default;

        CUDA_GENERAL_CALL Triangle(const T &_p1, const T &_p2, const T &_p3) : p1(_p1), p2(_p2), p3(_p3) {}
    };

    namespace gvis {

        // Helper function to write single vertex to OBJ file
        static void write_vertex(std::ofstream &output, const Eigen::Vector3d &v) {
            output << "v " << v.x() << " " << v.y() << " " << v.z() << std::endl;
        }

        // Helper function to write single vertex to OBJ file
        static void write_vertex(std::ofstream &output, const Eigen::Vector3d &v, const Eigen::Vector3d &rgb) {
            output << "v " << v.x() << " " << v.y() << " " << v.z() << " " << rgb.x() << " " << rgb.y() << " "
                   << rgb.z() << std::endl;
        }

        // Helper function to write single vertex to OBJ file
        static void write_vertex_to_xyz(std::ofstream &output, const Eigen::Vector3d &v) {
            output << v.x() << " " << v.y() << " " << v.z() << std::endl;
        }

        // Helper function to write face
        static void write_face(std::ofstream &output, const Eigen::Vector3i &f) {
            output << "f " << f.x() << " " << f.y() << " " << f.z() << std::endl;
        }

        // Helper function to write face
        static void write_face(std::ofstream &output, const Eigen::Vector4i &f) {
            output << "f " << f.x() << " " << f.y() << " " << f.z() << " " << f.w() << std::endl;
        }

        // Helper function to write line
        static void write_line(std::ofstream &output, const Eigen::Vector2i &l) {
            output << "l " << l.x() << " " << l.y() << std::endl;
        }

        // Helper function to write full cube (using relative vertex positions in the OBJ file - support for this should be widespread by now)
        inline void writeCube(const Eigen::Vector3d &nodeOrigin, const Eigen::Vector3d &unit, std::ofstream &output,
                              size_t &faceBegIdx) {
            //	   2-------1
            //	  /|      /|
            //	 / |     / |
            //	7--|----8  |
            //	|  4----|--3
            //	| /     | /
            //	5-------6
            // Create vertices
            Eigen::Vector3d v1 = nodeOrigin + Eigen::Vector3d(0, unit.y(), unit.z());
            Eigen::Vector3d v2 = nodeOrigin + Eigen::Vector3d(0, 0, unit.z());
            Eigen::Vector3d v3 = nodeOrigin + Eigen::Vector3d(0, unit.y(), 0);
            Eigen::Vector3d v4 = nodeOrigin;
            Eigen::Vector3d v5 = nodeOrigin + Eigen::Vector3d(unit.x(), 0, 0);
            Eigen::Vector3d v6 = nodeOrigin + Eigen::Vector3d(unit.x(), unit.y(), 0);
            Eigen::Vector3d v7 = nodeOrigin + Eigen::Vector3d(unit.x(), 0, unit.z());
            Eigen::Vector3d v8 = nodeOrigin + Eigen::Vector3d(unit.x(), unit.y(), unit.z());

            // write them in reverse order, so relative position is -i for v_i
            write_vertex(output, v1);
            write_vertex(output, v2);
            write_vertex(output, v3);
            write_vertex(output, v4);
            write_vertex(output, v5);
            write_vertex(output, v6);
            write_vertex(output, v7);
            write_vertex(output, v8);

            // create faces
#if defined(MESH_WRITE)
            // back
            write_face(output, Eigen::Vector3i(faceBegIdx + 1, faceBegIdx + 3, faceBegIdx + 4));
            write_face(output, Eigen::Vector3i(faceBegIdx + 1, faceBegIdx + 4, faceBegIdx + 2));
            // bottom
            write_face(output, Eigen::Vector3i(faceBegIdx + 4, faceBegIdx + 3, faceBegIdx + 6));
            write_face(output, Eigen::Vector3i(faceBegIdx + 4, faceBegIdx + 6, faceBegIdx + 5));
            // right
            write_face(output, Eigen::Vector3i(faceBegIdx + 3, faceBegIdx + 1, faceBegIdx + 8));
            write_face(output, Eigen::Vector3i(faceBegIdx + 3, faceBegIdx + 8, faceBegIdx + 6));
            // top
            write_face(output, Eigen::Vector3i(faceBegIdx + 1, faceBegIdx + 2, faceBegIdx + 7));
            write_face(output, Eigen::Vector3i(faceBegIdx + 1, faceBegIdx + 7, faceBegIdx + 8));
            // left
            write_face(output, Eigen::Vector3i(faceBegIdx + 2, faceBegIdx + 4, faceBegIdx + 5));
            write_face(output, Eigen::Vector3i(faceBegIdx + 2, faceBegIdx + 5, faceBegIdx + 7));
            // front
            write_face(output, Eigen::Vector3i(faceBegIdx + 5, faceBegIdx + 6, faceBegIdx + 8));
            write_face(output, Eigen::Vector3i(faceBegIdx + 5, faceBegIdx + 8, faceBegIdx + 7));
#  elif defined(CUBE_WRITE)
            // back
            write_face(output, Eigen::Vector4i(faceBegIdx + 3, faceBegIdx + 4, faceBegIdx + 2, faceBegIdx + 1));
            // bottom
            write_face(output, Eigen::Vector4i(faceBegIdx + 6, faceBegIdx + 5, faceBegIdx + 4, faceBegIdx + 3));
            // right
            write_face(output, Eigen::Vector4i(faceBegIdx + 1, faceBegIdx + 8, faceBegIdx + 6, faceBegIdx + 3));
            // top
            write_face(output, Eigen::Vector4i(faceBegIdx + 1, faceBegIdx + 2, faceBegIdx + 7, faceBegIdx + 8));
            // left
            write_face(output, Eigen::Vector4i(faceBegIdx + 4, faceBegIdx + 5, faceBegIdx + 7, faceBegIdx + 2));
            // front
            write_face(output, Eigen::Vector4i(faceBegIdx + 8, faceBegIdx + 7, faceBegIdx + 5, faceBegIdx + 6));
#  else
            write_line(output, Eigen::Vector2i(faceBegIdx + 1, faceBegIdx + 2));
            write_line(output, Eigen::Vector2i(faceBegIdx + 2, faceBegIdx + 7));
            write_line(output, Eigen::Vector2i(faceBegIdx + 7, faceBegIdx + 8));
            write_line(output, Eigen::Vector2i(faceBegIdx + 8, faceBegIdx + 1));

            write_line(output, Eigen::Vector2i(faceBegIdx + 3, faceBegIdx + 4));
            write_line(output, Eigen::Vector2i(faceBegIdx + 4, faceBegIdx + 5));
            write_line(output, Eigen::Vector2i(faceBegIdx + 5, faceBegIdx + 6));
            write_line(output, Eigen::Vector2i(faceBegIdx + 6, faceBegIdx + 3));

            write_line(output, Eigen::Vector2i(faceBegIdx + 3, faceBegIdx + 1));
            write_line(output, Eigen::Vector2i(faceBegIdx + 4, faceBegIdx + 2));
            write_line(output, Eigen::Vector2i(faceBegIdx + 5, faceBegIdx + 7));
            write_line(output, Eigen::Vector2i(faceBegIdx + 6, faceBegIdx + 8));
#endif

            faceBegIdx += 8;
        }

        inline void writePointCloud(const std::vector<Eigen::Vector3d> &points, std::ofstream &output) {
            for (size_t i = 0; i < points.size(); ++i)
                write_vertex(output, points[i]);
        }

        inline void writePointCloud_xyz(const std::vector<Eigen::Vector3d> &points, std::ofstream &output) {
            for (size_t i = 0; i < points.size(); ++i)
                write_vertex_to_xyz(output, points[i]);
        }

        inline void
        writePointCloud(const std::vector<Eigen::Vector3d> &points, const std::vector<Eigen::Vector3d> &rgbs,
                        std::ofstream &output) {
            for (size_t i = 0; i < points.size(); ++i)
                write_vertex(output, points[i], rgbs[i]);
        }

        inline void writePointCloud(const MXd &points, std::ofstream &output) {
            for (size_t i = 0; i < points.size(); ++i)
                write_vertex(output, points.row(i));
        }

        inline void writePointCloud_xyz(const MXd &points, std::ofstream &output) {
            for (size_t i = 0; i < points.rows(); ++i)
                write_vertex_to_xyz(output, points.row(i));
        }

        inline void
        writePointCloud(const MXd &points, const std::vector<Eigen::Vector3d> &rgbs, std::ofstream &output) {
            for (size_t i = 0; i < points.rows(); ++i)
                write_vertex(output, points.row(i), rgbs[i]);
        }

        inline void writePointCloud(const Eigen::Vector3d &point, const Eigen::Vector3d &rgb, std::ofstream &output) {
            write_vertex(output, point, rgb);
        }

    } // namespace gvis

    namespace pointgen {
        template<typename Scalar, int DIM, typename T=Eigen::Matrix<Scalar, DIM, 1>>
        inline bool getUniformRandomMatrix(const T &min_area,
                                           const T &max_area,
                                           size_t num, Eigen::Matrix<Scalar, Eigen::Dynamic, DIM> &M,
                                           double scaleFactor = 1.0, const T& dis = T::Zero()) {
            M.resize(num, DIM);
            std::default_random_engine e(time(0)); // current time as seed
            std::uniform_real_distribution<Scalar> n(-1, 1);
            M = Eigen::Matrix<Scalar, Eigen::Dynamic, DIM>::Zero(num, DIM)
                    .unaryExpr([&](Scalar val) { return static_cast<Scalar>(round((n(e) + 1e-6) * 1e6) / 1e6); });

            M.conservativeResize(num + 1, DIM + 1); // strange value because reallocate memory
            Eigen::Matrix<Scalar, 1, DIM + 1> r_zeroVec(num);
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> c_onesVec(num + 1);
            r_zeroVec.setZero();
            c_onesVec.setOnes();
            M.row(num) = r_zeroVec;
            M.col(DIM) = c_onesVec;

            const T min_m = M.colwise().minCoeff();
            const T max_m = M.colwise().maxCoeff();

            T diag_area = max_area - min_area; // bounding box
            T diag_m = max_m - min_m; // random points

            T center_area = 0.5 * (max_area + min_area);
            T center_m = 0.5 * (max_m + min_m);

            Eigen::Matrix<Scalar, DIM + 1, DIM + 1> zoomMatrix = Eigen::Matrix<Scalar, DIM + 1, DIM + 1>::Identity();
            Eigen::Matrix<Scalar, DIM + 1, DIM + 1> transMatrix = Eigen::Matrix<Scalar, DIM + 1, DIM + 1>::Identity();

            for (int d = 0; d < DIM; d++) {
                if (diag_m[d] == 0) return false;
                zoomMatrix(d, d) = (diag_area[d] / diag_m[d]) * scaleFactor;
                transMatrix(DIM, d) = (center_area[d] - center_m[d] * zoomMatrix(d, d)) + dis(d);
            }

            M = M * zoomMatrix * transMatrix;
            M = (M.array().abs() < 1e-9).select(0, M);

            return true;
        }

        template<typename Scalar, int DIM, typename T=Eigen::Matrix<Scalar, DIM, 1>>
        inline bool getGaussianRandomMatrix(const T &min_area,
                                            const T &max_area,
                                            size_t num, Eigen::Matrix<Scalar, Eigen::Dynamic, DIM> &M,
                                            double scaleFactor = 1.0, const T& dis = T::Zero()) {
            M.resize(num, DIM);
            std::default_random_engine e(1314); // current time as seed
            std::normal_distribution<Scalar> n(-1, 1);
            M = Eigen::Matrix<Scalar, Eigen::Dynamic, DIM>::Zero(num, DIM)
                    .unaryExpr([&](Scalar val) { return static_cast<Scalar>(round((n(e) + 1e-6) * 1e6) / 1e6); });

            M.conservativeResize(num + 1, DIM + 1); // strange value because reallocate memory
            Eigen::Matrix<Scalar, 1, DIM + 1> r_zeroVec(num);
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> c_onesVec(num + 1);
            r_zeroVec.setZero();
            c_onesVec.setOnes();
            M.row(num) = r_zeroVec;
            M.col(DIM) = c_onesVec;

            const T min_m = M.colwise().minCoeff();
            const T max_m = M.colwise().maxCoeff();

            T diag_area = max_area - min_area; // bounding box
            T diag_m = max_m - min_m; // random points

            T center_area = 0.5 * (max_area + min_area);
            T center_m = 0.5 * (max_m + min_m);

            Eigen::Matrix<Scalar, DIM + 1, DIM + 1> zoomMatrix = Eigen::Matrix<Scalar, DIM + 1, DIM + 1>::Identity();
            Eigen::Matrix<Scalar, DIM + 1, DIM + 1> transMatrix = Eigen::Matrix<Scalar, DIM + 1, DIM + 1>::Identity();

            for (int d = 0; d < DIM; d++) {
                if (diag_m[d] == 0) return false;
                zoomMatrix(d, d) = (diag_area[d] / diag_m[d]) * scaleFactor;
                transMatrix(DIM, d) = (center_area[d] - center_m[d] * zoomMatrix(d, d)) + dis(d);
            }

            M = M * zoomMatrix * transMatrix;
            M = (M.array().abs() < 1e-9).select(0, M);

            return true;
        }

        template<typename Scalar, int DIM, typename T=Eigen::Matrix<Scalar, DIM, 1>>
        inline bool genUniformRandomPoints(const AABBox<Scalar, DIM, T> &_area,
                                           size_t num, Eigen::Matrix<Scalar, Eigen::Dynamic, DIM> &M,
                                           double scaleFactor = 1.0, const T &dis = T::Zero()) {
            static_assert(DIM >= 2, "DIM must at least 2");
            if (scaleFactor <= .0) return false;

            AABBox<Scalar, DIM, T> area = _area;
            area.scaleAndTranslate(scaleFactor, dis);

            auto haltonSequence = [](int index, int base) {
                double result = 0.0;
                double f = 1.0 / base;
                int i = index;

                while (i > 0) {
                    result += f * (i % base);
                    i = std::floor(i / base);
                    f /= base;
                }

                return result;
            };

            auto mapToRange = [](double value, double min, double max) {
                return min + value * (max - min);
            };

            T minArea = area.boxOrigin;
            T maxArea = area.boxEnd;

            int baseX = 2;
            int baseY = 3;
            int baseZ;
            if constexpr (DIM == 3) baseZ = 5;

            M.resize(num, DIM);
            for (int i = 0; i < num; ++i) {
                Scalar x = mapToRange(haltonSequence(i, baseX), minArea.x(), maxArea.x());
                Scalar y = mapToRange(haltonSequence(i, baseY), minArea.y(), maxArea.y());
                if constexpr (DIM == 3) {
                    Scalar z = mapToRange(haltonSequence(i, baseZ), minArea.z(), maxArea.z());
                    M.row(i) = Eigen::Matrix<Scalar, 1, DIM>(x, y, z);
                } else {
                    M.row(i) = Eigen::Matrix<Scalar, 1, DIM>(x, y);
                }
            }

            return true;
        }

        template<typename Scalar, int DIM, typename T=Eigen::Matrix<Scalar, DIM, 1>>
        inline bool genGaussianRandomPoints(const AABBox<Scalar, DIM, T> &_area,
                                            size_t num, Eigen::Matrix<Scalar, Eigen::Dynamic, DIM> &M,
                                            double scaleFactor = 1.0, const T &dis = T::Zero()) {
            static_assert(DIM >= 2, "DIM must at least 2");
            if (scaleFactor <= .0) return false;

            AABBox<Scalar, DIM, T> area = _area;
            area.scaleAndTranslate(scaleFactor, dis);

            static std::random_device rd; //
            static std::mt19937 gen(rd());
            auto gaussianSample = [](Scalar mean, Scalar stddev) {
                std::normal_distribution<double> dist(0.0, 1.0);

                T sample;
                for (int i = 0; i < DIM; ++i)
                    sample(i) = mean(i) + stddev(i) * dist(gen);

                return sample;
            };

            T minArea = area.boxOrigin;
            T maxArea = area.boxEnd;

            T mean = (maxArea + minArea) / 2.0;
            T stddev = (maxArea - minArea) / 6.0;

            M.resize(num, DIM);
            for (int i = 0; i < num; ++i) {
                Eigen::Matrix<Scalar, 1, DIM> sample = gaussianSample(mean, stddev);
                M.row(i) = sample;
            }

            return true;
        }

    } // namespace pointgen

NAMESPACE_END(ITS)
