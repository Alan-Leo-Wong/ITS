#pragma once

#include "Config.hpp"
#include <fstream>
#include <Eigen/Dense>

NAMESPACE_BEGIN(ITS)
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
            Real center = (boxOrigin + boxEnd) / 2.0;

            Real scaled_min_point = (boxOrigin - center) * scale_factor + center + translation;
            Real scaled_max_point = (boxEnd - center) * scale_factor + center + translation;

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

        CUDA_GENERAL_CALL Triangle() = default;

        CUDA_GENERAL_CALL Triangle(const Real &_p1, const Real &_p2, const Real &_p3) : p1(_p1), p2(_p2), p3(_p3) {}
    };

    template<typename Scalar>
    inline bool getGaussianRandomMatrix(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &area,
                                        const size_t &num, const float &scaleFactor, const float &dis,
                                        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &M) {
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

    template<typename Scalar>
    inline bool getUniformRandomMatrix(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &area,
                                       const size_t &num, const float &scaleFactor, const float &dis,
                                       Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &M) {
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

    template<typename Scalar>
    inline bool getUniformRandomMatrix(const AABox<Scalar> &_area,
                                       const size_t &num, const double &scaleFactor, const V3d &dis,
                                       std::vector<Scalar> &randomPoints) {
        if (scaleFactor <= .0) return false;

        //using BoxType = AABox<Scalar>::type;

        AABox<Scalar> area = _area;
        area.scaleAndTranslate(scaleFactor, dis);

        // ����Halton���еĵ�index��ֵ
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

        // ��Halton����ֵӳ�䵽[min, max]��Χ��
        auto mapToRange = [](double value, double min, double max) {
            return min + value * (max - min);
        };

        // ��[minArea, maxArea]��Χ�ڽ�������������
        const Scalar &minArea = area.boxOrigin;
        const Scalar &maxArea = area.boxEnd;

        int baseX = 2; // X���ϵĻ���
        int baseY = 3; // Y���ϵĻ���
        int baseZ = 5; // Z���ϵĻ���

        for (int i = 0; i < num; ++i) {
            double x = mapToRange(haltonSequence(i, baseX), minArea.x(), maxArea.x());
            double y = mapToRange(haltonSequence(i, baseY), minArea.y(), maxArea.y());
            double z = mapToRange(haltonSequence(i, baseZ), minArea.z(), maxArea.z());
            randomPoints.emplace_back(Scalar(x, y, z));
        }

        return true;
    }

    template<typename Scalar>
    inline bool getUniformRandomMatrix(const AABox<Scalar> &_area,
                                       const size_t &num, const float &scaleFactor, const V3f &dis,
                                       Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &M) {
        std::vector<Scalar> randomPoints;
        if (getUniformRandomMatrix<Scalar>(_area, num, scaleFactor, dis, randomPoints)) {
            Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 3, Eigen::RowMajor>> mat(
                    reinterpret_cast<double *>(randomPoints.data()), num, 3);
            M = mat;
            return true;
        }
        return false;
    }

    template<typename Scalar>
    inline bool getGaussianRandomMatrix(const AABox<Scalar> &_area,
                                        const size_t &num, const double &scaleFactor, const V3d &dis,
                                        std::vector<Scalar> &randomPoints) {
        if (scaleFactor <= .0) return false;

        AABox<Scalar> area = _area;
        area.scaleAndTranslate(scaleFactor, dis);

        static std::random_device rd; // ����һ���������������
        // α�����������gen (������rd�������������Ϊstd::random_device���ܻ�����Ƚ������������)
        //static std::mt19937 gen(rd()); // ʹ�����������������������(ͨ��rd())���������Ӵ��ݸ�std::mt19937����gen���г�ʼ��
        static std::mt19937 gen(1314); // ʹ�ó������Ӵ��ݸ�std::mt19937����gen���г�ʼ��
        auto gaussianSample = [](const Scalar &mean, const Scalar &stddev) {
            std::normal_distribution<double> dist(0.0, 1.0);

            Scalar sample;
            for (int i = 0; i < 3; ++i)
                sample(i) = mean(i) + stddev(i) * dist(gen);

            return sample;
        };

        const Scalar &minArea = area.boxOrigin;
        const Scalar &maxArea = area.boxEnd;

        Scalar mean = (maxArea + minArea) / 2.0;
        Scalar stddev = (maxArea - minArea) / 6.0;

        for (int i = 0; i < num; ++i) {
            // ���ɸ�˹����
            Scalar sample = gaussianSample(mean, stddev);
            randomPoints.emplace_back(sample);
        }

        return true;
    }

    template<typename Scalar>
    inline bool getGaussianRandomMatrix(const Eigen::Matrix<Scalar, 3, 1> &min_area,
                                        const Eigen::Matrix<Scalar, 3, 1> &max_area, const size_t &num,
                                        const float &scaleFactor, const float &dis,
                                        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &M) {
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> _M(num, 3);
        std::default_random_engine e(1314); // current time as seed
        std::normal_distribution<Scalar> n(-1, 1);
        _M = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(num, 3)
                .unaryExpr([&](Scalar val) { return static_cast<Scalar>(round((n(e) + 1e-6) * 1e6) / 1e6); });

        _M.conservativeResize(num + 1, 4); // strange value because reallocate memory
        Eigen::Matrix<Scalar, 1, 4> r_zeroVec(num);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> c_onesVec(num + 1);
        r_zeroVec.setZero();
        c_onesVec.setOnes();
        _M.row(num) = r_zeroVec;
        _M.col(3) = c_onesVec;

        const auto min_m = _M.colwise().minCoeff();
        const auto max_m = _M.colwise().maxCoeff();

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

        _M = _M * zoomMatrix * transMatrix;
        M.resize(num, 3);
        M = _M.block(0, 0, num, 3);
        M = (M.array().abs() < 1e-9).select(0, M);

        return true;
    }

    template<typename Scalar>
    inline bool getUniformRandomMatrix(const Eigen::Matrix<Scalar, 3, 1> &min_area,
                                       const Eigen::Matrix<Scalar, 3, 1> &max_area, const size_t &num,
                                       const float &scaleFactor, const float &dis,
                                       Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &M) {
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> _M(num, 3);
        //std::default_random_engine e(time(0)); // current time as seed
        std::default_random_engine e(1314);
        std::uniform_real_distribution<Scalar> n(-1, 1);
        _M = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(num, 3)
                .unaryExpr([&](Scalar val) { return static_cast<Scalar>(round((n(e) + 1e-6) * 1e6) / 1e6); });

        _M.conservativeResize(num + 1, 4); // strange value because reallocate memory
        Eigen::Matrix<Scalar, 1, 4> r_zeroVec(num);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> c_onesVec(num + 1);
        r_zeroVec.setZero();
        c_onesVec.setOnes();
        _M.row(num) = r_zeroVec;
        _M.col(3) = c_onesVec;

        const auto min_m = _M.colwise().minCoeff();
        const auto max_m = _M.colwise().maxCoeff();

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

        _M = _M * zoomMatrix * transMatrix;
        M.resize(num, 3);
        M = _M.block(0, 0, num, 3);
        M = (M.array().abs() < 1e-9).select(0, M);

        return true;
    }

NAMESPACE_END(ITS)
