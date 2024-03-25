#include "ThinShells.hpp"
#include "BSpline.hpp"
#include "MortonLUT.hpp"
#include "CUDACompute.hpp"
#include "utils/IO.hpp"
#include "utils/Timer.hpp"
#include "utils/Common.hpp"
#include "utils/String.hpp"
#include "detail/cuda/CUDAMath.hpp"
#include "mc/MarchingCubes.hpp"
#include <omp.h>
#include <queue>
#include <iomanip>
#include <cassert>
#include <numeric>
#include <Eigen/Sparse>
#include <igl/signed_distance.h>

NAMESPACE_BEGIN(ITS)
    namespace core {

        //////////////////////
        //  Create  Shells  //
        //////////////////////
        int parallelAxis(const Vector3d &p1, const Vector3d &p2) {
            if (fabs(p1.y() - p2.y()) < 1e-9 && fabs(p1.z() - p2.z()) < 1e-9) return 1; // ��x��ƽ��
            else if (fabs(p1.x() - p2.x()) < 1e-9 && fabs(p1.z() - p2.z()) < 1e-9) return 2; // ��y��ƽ��
            else return 3; // ��z��ƽ��
        };

        Vector3i ThinShells::getPointDis(const Vector3d &vert, const Vector3d &origin, const Vector3d &width) const {
            return ((vert - origin).array() / width.array()).cast<int>();
        }

        Vector3i ThinShells::getPointDis(const Vector3d &vert, const Vector3d &origin, const double &width) const {
            return ((vert - origin).array() / width).cast<int>();
        }

        void ThinShells::cpIntersectionPoints() {
            vector<V2i> modelEdges = extractEdges();
            uint nModelEdges = modelEdges.size();

            const size_t &numFineNodes = svo.numFineNodes;
            std::cout << "-- Number of level-0 nodes: " << numFineNodes << std::endl;;

            const vector<SVONode> &nodeArray = svo.svoNodeArray;
            const vector<node_edge_type> &fineNodeEdges = svo.fineNodeEdgeArray;

            // ֻ��Ҫ������������ײ�ڵ�Ľ���

            // �����εı���node�潻�� ��Ϊ��ʽB��������������left/bottom/back corner�ϣ� ������ڵ�ֻ��Ҫ������������Ľ�����
            std::cout << "1. Computing the intersections between mesh EDGES and nodes...\n";

            TimerInterface *timer = nullptr;
            createTimer(&timer);
            startTimer(&timer);

            vector<size_t> fineNodeIdx(svo.numFineNodes);
            std::iota(fineNodeIdx.begin(), fineNodeIdx.end(), 0);
            std::transform(nodeArray.begin(), nodeArray.begin() + svo.numFineNodes,
                           fineNodeIdx.begin(), std::inserter(morton2FineNodeIdx, morton2FineNodeIdx.end()),
                           [](const SVONode &node, const size_t &idx) {
                               return std::make_pair(node.mortonCode, idx);
                           });

#pragma omp parallel
            for (int i = 0; i < nModelEdges; i++) {
                std::vector<Vector3d> edge_vec_private;
                //std::vector<std::pair<Vector3d, uint32_t>> edge_morton_vec_private;

                Eigen::Vector2i e = modelEdges[i];
                Vector3d p1 = vertMat.row(e.x());
                Vector3d p2 = vertMat.row(e.y());
                //if (!isLess(p1, p2, std::less<Vector3d>())) std::swap(p1, p2);

                Vector3d modelEdgeDir = p2 - p1;

                Vector3i dis1 = getPointDis(p1, modelOrigin, Vector3d(voxelWidth, voxelWidth, voxelWidth));
                Vector3i dis2 = getPointDis(p2, modelOrigin, Vector3d(voxelWidth, voxelWidth, voxelWidth));

                Vector3i min_dis = clamp(vmini(dis1, dis2).array() - 1, Vector3i(0, 0, 0), svo_gridSize.array() - 1);
                Vector3i max_dis = clamp(vmaxi(dis1, dis2).array() + 1, Vector3i(0, 0, 0), svo_gridSize.array() - 1);

                //#pragma omp for nowait collapse(3)
#pragma omp for nowait
                for (int z = min_dis.z(); z <= max_dis.z(); ++z) {
                    for (int y = min_dis.y(); y <= max_dis.y(); ++y) {
                        for (int x = min_dis.x(); x <= max_dis.x(); ++x) {
                            uint32_t nodeMorton = morton::mortonEncode_LUT((uint16_t) x, (uint16_t) y, (uint16_t) z);
                            if (morton2FineNodeIdx.find(nodeMorton) == morton2FineNodeIdx.end()) continue;
                            Vector3d lbbCorner = nodeArray[morton2FineNodeIdx.at(
                                    nodeMorton)].origin; // at() is thread safe
                            double width = nodeArray[morton2FineNodeIdx.at(nodeMorton)].width;

                            // back plane
                            double back_t = DINF;
                            if (modelEdgeDir.x() != 0) back_t = (lbbCorner.x() - p1.x()) / modelEdgeDir.x();
                            // left plane
                            double left_t = DINF;
                            if (modelEdgeDir.y() != 0) left_t = (lbbCorner.y() - p1.y()) / modelEdgeDir.y();
                            // bottom plane
                            double bottom_t = DINF;
                            if (modelEdgeDir.z() != 0) bottom_t = (lbbCorner.z() - p1.z()) / modelEdgeDir.z();

                            if (donut::isInRange(.0, 1.0, back_t) &&
                                donut::isInRange(lbbCorner.y(), lbbCorner.y() + width,
                                                 (p1 + back_t * modelEdgeDir).y()) &&
                                donut::isInRange(lbbCorner.z(), lbbCorner.z() + width,
                                                 (p1 + back_t * modelEdgeDir).z())) {
                                //edgeInterPoints.emplace_back(p1 + back_t * modelEdgeDir);
                                edge_vec_private.emplace_back(p1 + back_t * modelEdgeDir);
                                //edge_morton_vec_private.emplace_back(std::make_pair(p1 + back_t * modelEdgeDir, nodeMorton));
                            }
                            if (donut::isInRange(.0, 1.0, left_t) &&
                                donut::isInRange(lbbCorner.x(), lbbCorner.x() + width,
                                                 (p1 + left_t * modelEdgeDir).x()) &&
                                donut::isInRange(lbbCorner.z(), lbbCorner.z() + width,
                                                 (p1 + left_t * modelEdgeDir).z())) {
                                //edgeInterPoints.emplace_back(p1 + left_t * modelEdgeDir);
                                edge_vec_private.emplace_back(p1 + left_t * modelEdgeDir);
                                //edge_morton_vec_private.emplace_back(std::make_pair(p1 + back_t * modelEdgeDir, nodeMorton));
                            }
                            if (donut::isInRange(.0, 1.0, bottom_t) &&
                                donut::isInRange(lbbCorner.x(), lbbCorner.x() + width,
                                                 (p1 + bottom_t * modelEdgeDir).x()) &&
                                donut::isInRange(lbbCorner.y(), lbbCorner.y() + width,
                                                 (p1 + bottom_t * modelEdgeDir).y())) {
                                //edgeInterPoints.emplace_back(p1 + bottom_t * modelEdgeDir);
                                edge_vec_private.emplace_back(p1 + bottom_t * modelEdgeDir);
                                //edge_morton_vec_private.emplace_back(std::make_pair(p1 + back_t * modelEdgeDir, nodeMorton));
                            }
                        }
                    }
                }

#pragma omp critical
                {
                    edgeInterPoints.insert(edgeInterPoints.end(), edge_vec_private.begin(), edge_vec_private.end());
                }
            }

            std::sort(edgeInterPoints.begin(), edgeInterPoints.end(), std::less<Vector3d>());
            edgeInterPoints.erase(std::unique(edgeInterPoints.begin(), edgeInterPoints.end()), edgeInterPoints.end());
            allInterPoints.insert(allInterPoints.end(), edgeInterPoints.begin(), edgeInterPoints.end());

            stopTimer(&timer);
            double time = getElapsedTime(&timer) * 1e-3;
            test_time::test_allTime += time;

            std::cout << "-- �����α���node�Ľ���������" << edgeInterPoints.size() << std::endl;

            // ����������node���߽�
            std::cout << "2. Computing the intersections between mesh FACES and node EDGES..." << std::endl;

            resetTimer(&timer);

            startTimer(&timer);

            const int numFineNodeEdges = fineNodeEdges.size();
            vector<node_edge_type> t_fineNodeEdges(numFineNodeEdges);
            std::transform(fineNodeEdges.begin(), fineNodeEdges.end(), t_fineNodeEdges.begin(),
                           [](const node_edge_type &a) {
                               if (!isLess<Vector3d>(a.first.first, a.first.second, std::less<Vector3d>())) {
                                   node_edge_type _p;
                                   _p.first.first = a.first.second, _p.first.second = a.first.first;
                                   _p.second = a.second;
                                   return _p;
                               } else return a;
                           });

            // ��ĩ�˵��x�����С��������
            struct x_sortEdge {
                bool operator()(node_edge_type &a, node_edge_type &b) {
                    if (fabs(a.first.second.x() - b.first.second.x()) < 1e-9) // ��x�������(ʣ���������ĸ����ĸ�������ν��)
                    {
                        if (fabs(a.first.second.y() - b.first.second.y()) < 1e-9)  // ��x��y���궼���
                            return a.first.second.z() < b.first.second.z(); // ����zС���Ǹ�
                        else
                            return a.first.second.y() < b.first.second.y(); // ����yС���Ǹ�
                    } else return a.first.second.x() < b.first.second.x();
                }
            };
            // ��ĩ�˵��y�����С��������
            struct y_sortEdge {
                bool operator()(node_edge_type &a, node_edge_type &b) {
                    if (fabs(a.first.second.y() - b.first.second.y()) < 1e-9) // ��y�������
                    {
                        if (fabs(a.first.second.x() - b.first.second.x()) < 1e-9)  // ��y��x���궼���
                            return a.first.second.z() < b.first.second.z(); // ����zС���Ǹ�
                        else
                            return a.first.second.x() < b.first.second.x(); // ����xС���Ǹ�
                    } else return a.first.second.y() < b.first.second.y();
                }
            };
            // ��ĩ�˵��z�����С��������
            struct z_sortEdge {
                bool operator()(node_edge_type &a, node_edge_type &b) {
                    if (fabs(a.first.second.z() - b.first.second.z()) < 1e-9) // ��z�������
                    {
                        if (fabs(a.first.second.x() - b.first.second.x()) < 1e-9)  // ��z��x���궼���
                            return a.first.second.y() < b.first.second.y(); // ����yС���Ǹ�
                        else
                            return a.first.second.x() < b.first.second.x(); // ����xС���Ǹ�
                    } else return a.first.second.z() < b.first.second.z();
                }
            };

            std::vector<node_edge_type> x_fineNodeEdges;
            std::vector<node_edge_type> y_fineNodeEdges;
            std::vector<node_edge_type> z_fineNodeEdges;
            //#pragma omp parallel
            {
                std::copy_if(t_fineNodeEdges.begin(), t_fineNodeEdges.end(), std::back_inserter(x_fineNodeEdges),
                             [](const node_edge_type &val) {
                                 return parallelAxis(val.first.first, val.first.second) == 1;
                             });
                std::sort(x_fineNodeEdges.begin(), x_fineNodeEdges.end(), x_sortEdge());

                std::copy_if(t_fineNodeEdges.begin(), t_fineNodeEdges.end(), std::back_inserter(y_fineNodeEdges),
                             [](const node_edge_type &val) {
                                 return parallelAxis(val.first.first, val.first.second) == 2;
                             });
                std::sort(y_fineNodeEdges.begin(), y_fineNodeEdges.end(), y_sortEdge());

                std::copy_if(t_fineNodeEdges.begin(), t_fineNodeEdges.end(), std::back_inserter(z_fineNodeEdges),
                             [](const node_edge_type &val) {
                                 return parallelAxis(val.first.first, val.first.second) == 3;
                             });
                std::sort(z_fineNodeEdges.begin(), z_fineNodeEdges.end(), z_sortEdge());
            }

            struct lessXVal {
                bool operator()(const node_edge_type &a,
                                const node_edge_type &b) { // Search for first element 'a' in list such that b �� a
                    return donut::isLargeDouble(b.first.second.x(), a.first.second.x(), 1e-9);
                }
            };
            struct lessYVal {
                bool operator()(const node_edge_type &a,
                                const node_edge_type &b) { // Search for first element 'a' in list such that b �� a
                    return donut::isLargeDouble(b.first.second.y(), a.first.second.y(), 1e-9);
                }
            };
            struct lessZVal {
                bool operator()(const node_edge_type &a,
                                const node_edge_type &b) { // Search for first element 'a' in list such that b �� a
                    return donut::isLargeDouble(b.first.second.z(), a.first.second.z(), 1e-9);
                }
            };

            const size_t x_numEdges = x_fineNodeEdges.size();
            const size_t y_numEdges = y_fineNodeEdges.size();
            const size_t z_numEdges = z_fineNodeEdges.size();

#pragma omp parallel
            for (const auto &tri: trisVec) {
                Vector3d triEdge_1 = tri.p2 - tri.p1;
                Vector3d triEdge_2 = tri.p3 - tri.p2;
                Vector3d triEdge_3 = tri.p1 - tri.p3;
                Vector3d triNormal = tri.normal;
                double triDir = tri.dir;
                Vector3d tri_bbox_origin = fminf(tri.p1, fminf(tri.p2, tri.p3));
                Vector3d tri_bbox_end = fmaxf(tri.p1, fmaxf(tri.p2, tri.p3));

                // Search for first element x such that _q �� x
                node_edge_type x_q;
                x_q.first.second = Eigen::Vector3d(tri_bbox_origin.x(), 0, 0);
                auto x_lower = std::lower_bound(x_fineNodeEdges.begin(), x_fineNodeEdges.end(), x_q, lessXVal());;
                if (x_lower != x_fineNodeEdges.end()) {
                    std::vector<Vector3d> x_face_vec_private;

#pragma omp for nowait
                    for (int i = x_lower - x_fineNodeEdges.begin(); i < x_numEdges; ++i) {
                        auto e_p1 = x_fineNodeEdges[i].first.first, e_p2 = x_fineNodeEdges[i].first.second;

                        if (donut::isLargeDouble(e_p1.x(), tri_bbox_end.x(), 1e-9)) break; // ��ʼ�˵����bbox_end

                        Vector3d edgeDir = e_p2 - e_p1;

                        if (fabsf(triNormal.dot(edgeDir)) < 1e-9) continue;

                        double t = (-triDir - triNormal.dot(e_p1)) / (triNormal.dot(edgeDir));
                        if (t < 0. || t > 1.) continue;
                        Vector3d interPoint = e_p1 + edgeDir * t;

                        if (triEdge_1.cross(interPoint - tri.p1).dot(triNormal) < 0) continue;
                        if (triEdge_2.cross(interPoint - tri.p2).dot(triNormal) < 0) continue;
                        if (triEdge_3.cross(interPoint - tri.p3).dot(triNormal) < 0) continue;

                        x_face_vec_private.emplace_back(interPoint);
                    }
                    if (!x_face_vec_private.empty()) {
#pragma omp critical
                        {
                            faceInterPoints.insert(faceInterPoints.end(), x_face_vec_private.begin(),
                                                   x_face_vec_private.end());
                        }
                    }
                }

                node_edge_type y_q;
                y_q.first.second = Eigen::Vector3d(0, tri_bbox_origin.y(), 0);
                // Search for first element x such that _q �� x
                auto y_lower = std::lower_bound(y_fineNodeEdges.begin(), y_fineNodeEdges.end(), y_q, lessYVal());
                if (y_lower != y_fineNodeEdges.end()) {
                    std::vector<Vector3d> y_face_vec_private;

#pragma omp for nowait
                    for (int i = y_lower - y_fineNodeEdges.begin(); i < y_numEdges; ++i) {
                        auto e_p1 = y_fineNodeEdges[i].first.first, e_p2 = y_fineNodeEdges[i].first.second;

                        if (donut::isLargeDouble(e_p1.y(), tri_bbox_end.y(), 1e-9)) break; // ��ʼ�˵����bbox_end

                        Vector3d edgeDir = e_p2 - e_p1;

                        if (fabsf(triNormal.dot(edgeDir)) < 1e-9) continue;

                        double t = (-triDir - triNormal.dot(e_p1)) / (triNormal.dot(edgeDir));
                        if (t < 0. || t > 1.) continue;
                        Vector3d interPoint = e_p1 + edgeDir * t;

                        if (triEdge_1.cross(interPoint - tri.p1).dot(triNormal) < 0) continue;
                        if (triEdge_2.cross(interPoint - tri.p2).dot(triNormal) < 0) continue;
                        if (triEdge_3.cross(interPoint - tri.p3).dot(triNormal) < 0) continue;

                        y_face_vec_private.emplace_back(interPoint);
                    }
                    if (!y_face_vec_private.empty()) {
#pragma omp critical
                        {
                            faceInterPoints.insert(faceInterPoints.end(), y_face_vec_private.begin(),
                                                   y_face_vec_private.end());
                        }
                    }
                }

                // Search for first element x such that _q �� x
                node_edge_type z_q;
                z_q.first.second = Eigen::Vector3d(0, 0, tri_bbox_origin.z());
                auto z_lower = std::lower_bound(z_fineNodeEdges.begin(), z_fineNodeEdges.end(), z_q, lessZVal());
                if (z_lower != z_fineNodeEdges.end()) {
                    std::vector<Vector3d> z_face_vec_private;

#pragma omp for nowait
                    for (int i = z_lower - z_fineNodeEdges.begin(); i < z_numEdges; ++i) {
                        auto e_p1 = z_fineNodeEdges[i].first.first, e_p2 = z_fineNodeEdges[i].first.second;

                        if (donut::isLargeDouble(e_p1.z(), tri_bbox_end.z(), 1e-9)) break; // ��ʼ�˵����bbox_end

                        Vector3d edgeDir = e_p2 - e_p1;

                        if (fabsf(triNormal.dot(edgeDir)) < 1e-9) continue;

                        double t = (-triDir - triNormal.dot(e_p1)) / (triNormal.dot(edgeDir));
                        if (t < 0. || t > 1.) continue;
                        Vector3d interPoint = e_p1 + edgeDir * t;

                        if (triEdge_1.cross(interPoint - tri.p1).dot(triNormal) < 0) continue;
                        if (triEdge_2.cross(interPoint - tri.p2).dot(triNormal) < 0) continue;
                        if (triEdge_3.cross(interPoint - tri.p3).dot(triNormal) < 0) continue;

                        z_face_vec_private.emplace_back(interPoint);
                    }
                    if (!z_face_vec_private.empty()) {
#pragma omp critical
                        {
                            faceInterPoints.insert(faceInterPoints.end(), z_face_vec_private.begin(),
                                                   z_face_vec_private.end());
                        }
                    }
                }
            }

            //#pragma omp parallel
            //	for (const auto& tri : modelTris)
            //	{
            //		std::vector<Vector3d> face_vec_private;
            //		Vector3d triEdge_1 = tri.p2 - tri.p1; Vector3d triEdge_2 = tri.p3 - tri.p2; Vector3d triEdge_3 = tri.p1 - tri.p3;
            //		Vector3d triNormal = tri.normal; double triDir = tri.dir;
            //		Vector3d tri_bbox_origin = fminf(tri.p1, fminf(tri.p2, tri.p3));
            //		Vector3d tri_bbox_end = fmaxf(tri.p1, fmaxf(tri.p2, tri.p3));
            //
            //		double d_eps = 1e-9;
            //#pragma omp for nowait
            //		for (int j = 0; j < numFineNodeEdges; ++j)
            //		{
            //			auto nodeEdge = fineNodeEdges[j];
            //
            //			auto e_p1 = nodeEdge.first.first, e_p2 = nodeEdge.first.second;
            //			if (!isLess(e_p1, e_p2, std::less<Vector3d>())) std::swap(e_p1, e_p2);
            //
            //			const int parallel = parallelAxis(e_p1, e_p2);
            //			if (parallel == 1) // ��x��ƽ��(���˵�y��zֵ���)
            //			{
            //				if (isLessDouble(e_p1.y(), tri_bbox_origin.y(), d_eps) || isLargeDouble(e_p1.y(), tri_bbox_end.y(), d_eps) ||
            //					isLessDouble(e_p1.z(), tri_bbox_origin.z(), d_eps) || isLargeDouble(e_p1.z(), tri_bbox_end.z(), d_eps) ||
            //					isLessDouble(e_p2.x(), tri_bbox_origin.x(), d_eps) || isLargeDouble(e_p1.x(), tri_bbox_end.x(), d_eps)) continue;
            //			}
            //			else if (parallel == 2) // ��y��ƽ��
            //			{
            //				if (isLessDouble(e_p1.x(), tri_bbox_origin.x(), d_eps) || isLargeDouble(e_p1.x(), tri_bbox_end.x(), d_eps) ||
            //					isLessDouble(e_p1.z(), tri_bbox_origin.z(), d_eps) || isLargeDouble(e_p1.z(), tri_bbox_end.z(), d_eps) ||
            //					isLessDouble(e_p2.y(), tri_bbox_origin.y(), d_eps) || isLargeDouble(e_p1.y(), tri_bbox_end.y(), d_eps)) continue;
            //			}
            //			else // ��z��ƽ��
            //			{
            //				if (isLessDouble(e_p1.x(), tri_bbox_origin.x(), d_eps) || isLargeDouble(e_p1.x(), tri_bbox_end.x(), d_eps) ||
            //					isLessDouble(e_p1.y(), tri_bbox_origin.y(), d_eps) || isLargeDouble(e_p1.y(), tri_bbox_end.y(), d_eps) ||
            //					isLessDouble(e_p2.z(), tri_bbox_origin.z(), d_eps) || isLargeDouble(e_p1.z(), tri_bbox_end.z(), d_eps)) continue;
            //			}
            //
            //			Vector3d edgeDir = e_p2 - e_p1;
            //
            //			if (fabsf(triNormal.dot(edgeDir)) < 1e-9) continue;
            //
            //			double t = (-triDir - triNormal.dot(e_p1)) / (triNormal.dot(edgeDir));
            //			if (t < 0. || t > 1.) continue;
            //			Vector3d interPoint = e_p1 + edgeDir * t;
            //
            //			if (triEdge_1.cross(interPoint - tri.p1).dot(triNormal) < 0) continue;
            //			if (triEdge_2.cross(interPoint - tri.p2).dot(triNormal) < 0) continue;
            //			if (triEdge_3.cross(interPoint - tri.p3).dot(triNormal) < 0) continue;
            //
            //			face_vec_private.emplace_back(interPoint);
            //		}
            //
            //#pragma omp critical
            //		{
            //			faceInterPoints.insert(faceInterPoints.end(), face_vec_private.begin(), face_vec_private.end());
            //		}
            //	}

            //#pragma omp parallel
            //	for (const auto& tri : modelTris)
            //	{
            //		std::vector<Vector3d> face_vec_private;
            //
            //		Vector3d triEdge_1 = tri.p2 - tri.p1; Vector3d triEdge_2 = tri.p3 - tri.p2; Vector3d triEdge_3 = tri.p1 - tri.p3;
            //		Vector3d triNormal = tri.normal; double triDir = tri.dir;
            //
            //#pragma omp for nowait
            //		for (int j = 0; j < numFineNodeEdges; ++j)
            //		{
            //			const auto& nodeEdge = fineNodeEdges[j];
            //			thrust_edge_type edge = nodeEdge.first;
            //			Vector3d edgeDir = edge.second - edge.first;
            //
            //			if (fabsf(triNormal.dot(edgeDir)) < 1e-9) continue;
            //
            //			double t = (-triDir - triNormal.dot(edge.first)) / (triNormal.dot(edgeDir));
            //			if (t < 0. || t > 1.) continue;
            //			Vector3d interPoint = edge.first + edgeDir * t;
            //
            //			if (triEdge_1.cross(interPoint - tri.p1).dot(triNormal) < 0) continue;
            //			if (triEdge_2.cross(interPoint - tri.p2).dot(triNormal) < 0) continue;
            //			if (triEdge_3.cross(interPoint - tri.p3).dot(triNormal) < 0) continue;
            //
            //			//faceInterPoints.emplace_back(interPoint);
            //			face_vec_private.emplace_back(interPoint);
            //		}
            //
            //#pragma omp critical
            //		{
            //			faceInterPoints.insert(faceInterPoints.end(), face_vec_private.begin(), face_vec_private.end());
            //		}
            //	}

            allInterPoints.insert(allInterPoints.end(), faceInterPoints.begin(), faceInterPoints.end());

            stopTimer(&timer);
            time = getElapsedTime(&timer) * 1e-3;
            test_time::test_allTime += time;

            std::cout << "-- ����������node�ߵĽ���������" << faceInterPoints.size() << std::endl;
            std::cout << "-- �ܽ���������" << allInterPoints.size() << std::endl;

            deleteTimer(&timer);

#ifdef IO_SAVE
            saveLatentPoint("");
#endif
        }

        void ThinShells::cpSDFOfTreeNodes() {
            TimerInterface *timer = nullptr;
            createTimer(&timer);
            startTimer(&timer);

            const auto &depthNodeVertexArray = svo.depthNodeVertexArray;
            const auto &esumDepthNodeVerts = svo.esumDepthNodeVerts;
            const size_t &numNodeVerts = svo.numNodeVerts;
            MatrixXd pointsMat(numNodeVerts, 3);
            for (int d = 0; d < treeDepth; ++d) {
                const size_t &d_numNodeVerts = depthNodeVertexArray[d].size();
                for (int i = 0; i < d_numNodeVerts; ++i)
                    pointsMat.row(esumDepthNodeVerts[d] + i) = depthNodeVertexArray[d][i].first;
            }

            {
                VectorXi I;
                MatrixXd C, N;
                igl::signed_distance(pointsMat, vertMat, faceMat,
                                     igl::SignedDistanceType::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER, sdfVal, I, C,
                                     N);
            }

            stopTimer(&timer);
            double time = getElapsedTime(&timer) * 1e-3;
            test_time::test_allTime += time;

            deleteTimer(&timer);
        }

        void ThinShells::cpCoefficients() {
            using SpMat = Eigen::SparseMatrix<double>;
            using Trip = Eigen::Triplet<double>;

            const auto &nodeArray = svo.svoNodeArray;
            const vector<vector<node_vertex_type>> &depthNodeVertexArray = svo.depthNodeVertexArray;
            const vector<size_t> &esumDepthNodeVerts = svo.esumDepthNodeVerts;
            depthVert2Idx.resize(treeDepth);

            TimerInterface *timer = nullptr;
            createTimer(&timer);
            startTimer(&timer);

            // ÿ�㶥�㵽�����±�(ȫ��)��ӳ��
            for (int d = 0; d < treeDepth; ++d) {
                const size_t d_numNodeVerts = depthNodeVertexArray[d].size();
                vector<size_t> d_nodeVertexIdx(d_numNodeVerts);
                std::iota(d_nodeVertexIdx.begin(), d_nodeVertexIdx.end(), 0);

                const size_t &d_esumNodeVerts = esumDepthNodeVerts[d]; // ����������exclusive scan
                std::transform(depthNodeVertexArray[d].begin(), depthNodeVertexArray[d].end(),
                               d_nodeVertexIdx.begin(), std::inserter(depthVert2Idx[d], depthVert2Idx[d].end()),
                               [d_esumNodeVerts](const node_vertex_type &val, const size_t &idx) {
                                   return std::make_pair(val.first, d_esumNodeVerts + idx);
                               });
            }

            /*for (int d = 0; d < treeDepth; ++d)
            {
                const size_t d_numNodeVerts = depthNodeVertexArray[d].size();
                vector<size_t> d_nodeVertexIdx(d_numNodeVerts);
                std::iota(d_nodeVertexIdx.begin(), d_nodeVertexIdx.end(), 0);

                std::transform(depthNodeVertexArray[d].begin(), depthNodeVertexArray[d].end(), d_nodeVertexIdx.begin(),
                    std::inserter(depthVert2Idx[d], depthVert2Idx[d].end()),
                    [](const node_vertex_type& val, size_t i) {
                        return std::make_pair(val.first, i);
                    });
            }*/
            // initial matrix
            const size_t &numNodeVerts = svo.numNodeVerts;
            vector<Trip> matApVal;

#pragma omp parallel
            for (int d = 0; d < treeDepth; ++d) {
                const size_t d_numNodeVerts = depthNodeVertexArray[d].size(); // ÿ��ڵ�Ķ�������
                //const size_t& d_esumNodeVerts = esumDepthNodeVerts[d]; // ����������exclusive scan
#pragma omp for nowait
                for (int i = 0; i < d_numNodeVerts; ++i) {
                    Vector3d i_nodeVertex = depthNodeVertexArray[d][i].first;
                    uint32_t i_fromNodeIdx = depthNodeVertexArray[d][i].second;
                    const size_t rowIdx = depthVert2Idx[d].at(i_nodeVertex);

                    vector<Trip> private_matApVal;

                    //matApVal.emplace_back(Trip(d_esumNodeVerts + i, d_esumNodeVerts + i, 1)); // self
                    private_matApVal.emplace_back(Trip(rowIdx, rowIdx, 1)); // self

                    //#pragma omp parallel for
                    for (int j = d - 1; j >= 0; --j) {
                        if (depthVert2Idx[j].find(i_nodeVertex) == depthVert2Idx[j].end()) break;
                        //matApVal.emplace_back(Trip(d_esumNodeVerts + i, esumDepthNodeVerts[j] + nodeVertex2Idx[j][i_nodeVertex], 1)); // child
                        private_matApVal.emplace_back(Trip(rowIdx, depthVert2Idx[j].at(i_nodeVertex), 1)); // child
                    }

                    // parent
                    auto inDmPointTraits = svo.setInDomainPoints(nodeArray[i_fromNodeIdx].parent, d + 1, depthVert2Idx);
                    //auto inDmPointTraits = svo.setInDomainPoints(i_fromNodeIdx, d + 1, depthVert2Idx);
                    const int nInDmPoints = inDmPointTraits.size();

                    //#pragma omp parallel for
                    for (int k = 0; k < nInDmPoints; ++k) {
                        const auto inDmPointTrait = inDmPointTraits[k];
                        double val = BaseFunction4Point(std::get<0>(inDmPointTrait), std::get<1>(inDmPointTrait),
                                                        i_nodeVertex);
                        // assert(inDmPointsIdx[k] < numNodeVerts, "index of col > numNodeVertex!");
                        //if (val != 0) matApVal.emplace_back(Trip(d_esumNodeVerts + i, inDmPointsIdx[k], val));
                        if (val != 0) private_matApVal.emplace_back(Trip(rowIdx, std::get<2>(inDmPointTrait), val));
                        //if (val != 0)  private_matApVal.emplace_back(Trip(d_esumNodeVerts + i, inDmPointsIdx[k], val));
                    }

#pragma omp critical
                    {
                        matApVal.insert(matApVal.end(), private_matApVal.begin(), private_matApVal.end());
                    }
                }
            }

            SpMat A(numNodeVerts, numNodeVerts); // Ap
            A.setFromTriplets(matApVal.begin(), matApVal.end());
            auto b = sdfVal;

            stopTimer(&timer);
            double time = getElapsedTime(&timer) * 1e-3;
            test_time::test_allTime += time;

            resetTimer(&timer);

            startTimer(&timer);
            Eigen::LeastSquaresConjugateGradient<SpMat> lscg;
            lscg.compute(A);
            lambda = lscg.solve(b);

            stopTimer(&timer);
            time = getElapsedTime(&timer) * 1e-3;
            test_time::test_allTime += time;

            printf("-- Solve equation elapsed time: %lf s.\n", time);
            deleteTimer(&timer);

            std::cout << "-- Residual Error: " << (A * lambda - b).norm() << std::endl;

            //saveCoefficients(concatFilePath((string)OUT_DIR, modelName, std::to_string(maxDepth), (string)"Coefficients.txt"));
            /*for (int u = 0; u < treeDepth; ++u)
            {
                const size_t u_numNodeVerts = depthNodeVertexArray[u].size();
                const size_t& u_esumNodeVerts = esumDepthNodeVerts[u];
                for (int k = 0; k < u_numNodeVerts; ++k)
                {
                    double bSplineVal = 0, sdf = sdfVal[u_esumNodeVerts + k];
                    Vector3d vert = depthNodeVertexArray[u][k].first;
                    for (int d = 0; d < treeDepth; ++d)
                    {
                        const size_t d_numNodeVerts = depthNodeVertexArray[d].size();
                        const size_t& d_esumNodeVerts = esumDepthNodeVerts[d];
                        for (int j = 0; j < d_numNodeVerts; ++j)
                        {
                            Vector3d nodeVert = depthNodeVertexArray[d][j].first;
                            uint32_t nodeIdx = depthNodeVertexArray[d][j].second;
                            bSplineVal += lambda[d_esumNodeVerts + j] * (BaseFunction4Point(nodeVert, svo.svoNodeArray[nodeIdx].width, vert));
                        }
                    }

                    if (fabs(bSplineVal - sdf) > 1e-9)
                        printf("vertIdx = %llu, bSplineVal = %.10lf, sdf = %.10lf\n", u_esumNodeVerts + k, bSplineVal, sdf);
                }
            }*/
        }

        void ThinShells::cpLatentBSplineValue() {
            TimerInterface *timer = nullptr;
            createTimer(&timer);
            startTimer(&timer);

            const uint numAllPoints = nModelVerts + allInterPoints.size();
            std::vector<Vector3d> pointsData;
            pointsData.insert(pointsData.end(), vertVec.begin(), vertVec.end());
            pointsData.insert(pointsData.end(), allInterPoints.begin(), allInterPoints.end());
            const auto &svoNodeArray = svo.svoNodeArray;

            if (nodeWidthArray.empty()) {
                std::transform(svoNodeArray.begin(), svoNodeArray.end(), std::back_inserter(nodeWidthArray),
                               [](SVONode node) {
                                   return Eigen::Vector3d(node.width, node.width, node.width);
                               });
            }

            const uint nInterPoints = allInterPoints.size();
            bSplineVal.resize(numAllPoints);
            bSplineVal.setZero();

            /*cuAcc::cpBSplineVal(numAllPoints, svo.numNodeVerts, svo.numTreeNodes, pointsData,
                svo.nodeVertexArray, nodeWidthArray, lambda, bSplineVal);*/

#pragma omp parallel
            for (size_t i = 0; i < numAllPoints; ++i) {
                const Vector3d &point = pointsData[i];

                double sum = 0.0;
                Vector3i dis = getPointDis(point, modelOrigin, voxelWidth);
                const uint32_t pointMorton = morton::mortonEncode_LUT((uint16_t) dis.x(), (uint16_t) dis.y(),
                                                                      (uint16_t) dis.z());
                const uint32_t nodeIdx = morton2FineNodeIdx.at(pointMorton);

                auto inDmPointsTraits = svo.setInDomainPoints(nodeIdx, 0, depthVert2Idx);
                const int nInDmPointsTraits = inDmPointsTraits.size();

#pragma omp parallel for reduction(+ : sum) // forѭ���еı�����������з�������
                for (int j = 0; j < nInDmPointsTraits; ++j) {
                    const auto &inDmPointTrait = inDmPointsTraits[j];
                    sum += lambda[std::get<2>(inDmPointTrait)] *
                           BaseFunction4Point(std::get<0>(inDmPointTrait), std::get<1>(inDmPointTrait), point);
                }
                bSplineVal[i] = sum;
            }

            // --CPU--
            /*const vector<SVONode>& svoNodeArray = svo.svoNodeArray;
            const vector<vector<node_vertex_type>>& depthNodeVertexArray = svo.depthNodeVertexArray;
            const vector<size_t>& esumDepthNodeVerts = svo.esumDepthNodeVerts;

            for (int i = 0; i < nModelVerts; ++i)
            {
                const Vector3d& modelVert = modelVerts[i];
                for (int d = 0; d < treeDepth; ++d)
                {
                    const size_t d_numNodeVerts = depthNodeVertexArray[d].size();
                    const size_t& d_esumNodeVerts = esumDepthNodeVerts[d];
                    for (int j = 0; j < d_numNodeVerts; ++j)
                    {
                        Vector3d nodeVert = depthNodeVertexArray[d][j].first;
                        uint32_t nodeIdx = depthNodeVertexArray[d][j].second;
                        bSplineVal[i] += lambda[d_esumNodeVerts + j] * (BaseFunction4Point(nodeVert, svoNodeArray[nodeIdx].width, modelVert));
                    }
                }
            }

            int cnt = 0;
            for (int i = 0; i < nInterPoints; ++i)
            {
                cnt = i + nModelVerts;
                const Vector3d& interPoint = allInterPoints[i];
                for (int d = 0; d < treeDepth; ++d)
                {
                    const size_t d_numNodeVerts = depthNodeVertexArray[d].size();
                    const size_t& d_esumNodeVerts = esumDepthNodeVerts[d];
                    for (int j = 0; j < d_numNodeVerts; ++j)
                    {
                        Vector3d nodeVert = depthNodeVertexArray[d][j].first;
                        uint32_t nodeIdx = depthNodeVertexArray[d][j].second;
                        bSplineVal[cnt] += lambda[d_esumNodeVerts + j] * (BaseFunction4Point(nodeVert, svoNodeArray[nodeIdx].width, interPoint));
                    }
                }
            }*/

            innerShellIsoVal = *(std::min_element(bSplineVal.begin(), bSplineVal.end()));
            outerShellIsoVal = *(std::max_element(bSplineVal.begin(), bSplineVal.end()));

            stopTimer(&timer);
            double time = getElapsedTime(&timer) * 1e-3;
            test_time::test_allTime += time;

            deleteTimer(&timer);

            std::cout << "-- innerShellIsoVal: " << innerShellIsoVal << std::endl;
            std::cout << "-- outerShellIsoVal: " << outerShellIsoVal << std::endl;
            std::cout << "-- Thickness: " << std::setprecision(3) << (outerShellIsoVal - innerShellIsoVal) << std::endl;
        }

        void ThinShells::initBSplineTree() {
            TimerInterface *timer = nullptr;
            createTimer(&timer);

            std::cout << "\nComputing intersection points of " << std::quoted(modelName)
                      << "and level-0 nodes...\n=====================" << std::endl;
            startTimer(&timer);
            cpIntersectionPoints();
            stopTimer(&timer);
            double time = getElapsedTime(&timer) * 1e-3;
            //qp::qp_ctrl(tColor::GREEN);
            //qp::qprint("-- Elapsed time: ", time, "s.");
            printf("-- Elapsed time: %lf s.\n", time);
            //qp::qp_ctrl();
            std::cout << "=====================\n";
#ifdef IO_SAVE
            saveIntersections("", "");
#endif // IO_SAVE

            std::cout << "\nComputing discrete SDF of tree nodes..." << std::endl;
            startTimer(&timer);
            cpSDFOfTreeNodes();
            stopTimer(&timer);
            time = getElapsedTime(&timer) * 1e-3;
            printf("-- Elapsed time: %lf s.\n", time);
            std::cout << "=====================\n";
#ifdef IO_SAVE
            saveSDFValue("");
#endif // IO_SAVE

            std::cout << "\nComputing coefficients..." << std::endl;
            startTimer(&timer);
            cpCoefficients();
            stopTimer(&timer);
            time = getElapsedTime(&timer) * 1e-3;
            printf("-- Elapsed time: %lf s.\n", time);
            std::cout << "=====================\n";
#ifdef IO_SAVE
            saveCoefficients("");
#endif // IO_SAVE

            std::cout << "\nComputing B-Spline value..." << std::endl;
            startTimer(&timer);
            cpLatentBSplineValue();
            stopTimer(&timer);
            time = getElapsedTime(&timer) * 1e-3;
            printf("-- Elapsed time: %lf s.\n", time);
            std::cout << "=====================\n";
            //#ifdef IO_SAVE
            saveBSplineValue("");
            //#endif // IO_SAVE

            deleteTimer(&timer);
        }

        void ThinShells::creatShell() {
            initBSplineTree();
        }

        void ThinShells::singlePointQuery(const std::string &out_file, const Vector3d &point) {
            if (innerShellIsoVal == -DINF || outerShellIsoVal == -DINF) {
                printf("Error: You must create shells first!");
                return;
            }
            if (nodeWidthArray.empty()) {
                auto &svoNodeArray = svo.svoNodeArray;
                std::transform(svoNodeArray.begin(), svoNodeArray.end(), std::back_inserter(nodeWidthArray),
                               [](SVONode node) {
                                   return Eigen::Vector3d(node.width, node.width, node.width);
                               });
            }

            double q_bSplineVal;
            Vector3d rgb;
            cuAcc::cpBSplineVal(svo.numNodeVerts, svo.numTreeNodes, point, svo.nodeVertexArray, nodeWidthArray, lambda,
                                q_bSplineVal);
            if (innerShellIsoVal < q_bSplineVal && q_bSplineVal < outerShellIsoVal)
                rgb = Vector3d(0.56471, 0.93333, 0.56471);
            else rgb = Vector3d(1, 0.27059, 0);

            string _out_file = out_file;
            if (getFileExtension(_out_file) != ".obj")
                _out_file = (string) getDirName(out_file.c_str()) +
                            (string) getFileName(out_file.c_str()) + (string) ".obj";

            checkDir(_out_file);
            std::ofstream out(_out_file);
            if (!out) {
                fprintf(stderr, "[I/O] Error: File %s could not be opened!", _out_file.c_str());
                return;
            }
            std::cout << "-- Save query result to " << std::quoted(_out_file) <<
                      "-- [RED] point not on the surface, [GREEN] point lie on the surface" << std::endl;

            gvis::writePointCloud(point, rgb, out);
        }

        void ThinShells::multiPointQuery(const std::string &out_file, const vector<Vector3d> &points) {
            if (innerShellIsoVal == -DINF || outerShellIsoVal == -DINF) {
                printf("Error: You must create shells first!");
                return;
            }
            if (nodeWidthArray.empty()) {
                auto &svoNodeArray = svo.svoNodeArray;
                std::transform(svoNodeArray.begin(), svoNodeArray.end(), std::back_inserter(nodeWidthArray),
                               [](SVONode node) {
                                   return Eigen::Vector3d(node.width, node.width, node.width);
                               });
            }

            VectorXd q_bSplineVal;
            vector<Vector3d> rgbs;
            cuAcc::cpBSplineVal(points.size(), svo.numNodeVerts, svo.numTreeNodes,
                                points, svo.nodeVertexArray, nodeWidthArray, lambda, q_bSplineVal);
            std::transform(q_bSplineVal.begin(), q_bSplineVal.end(), std::back_inserter(rgbs),
                           [=](double val) {
                               Vector3d _t;
                               if (innerShellIsoVal < val && val < outerShellIsoVal)
                                   _t = Vector3d(0.56471, 0.93333, 0.56471);
                               else _t = Vector3d(1, 0.27059, 0);
                               return _t;
                           });

            string _out_file = out_file;
            if (getFileExtension(_out_file) != ".obj")
                _out_file = (string) getDirName(out_file.c_str()) +
                            (string) getFileName(out_file.c_str()) + (string) ".obj";

            checkDir(_out_file);
            std::ofstream out(_out_file);
            if (!out) {
                fprintf(stderr, "[I/O] Error: File %s could not be opened!", _out_file.c_str());
                return;
            }
            std::cout << "-- Save query result to " << std::quoted(_out_file) <<
                      "-- [RED] points not on the surface, [GREEN] points lie on the surface" << std::endl;

            gvis::writePointCloud(points, rgbs, out);
        }

        void ThinShells::multiPointQuery(const std::string &out_file, const MatrixXd &pointsMat) {
            if (innerShellIsoVal == -DINF || outerShellIsoVal == -DINF) {
                printf("Error: You must create shells first!");
                return;
            }

            vector<Vector3d> points;
            for (int i = 0; i < pointsMat.rows(); ++i) points.emplace_back(pointsMat.row(i));

            multiPointQuery(out_file, points);
        }

        //////////////////////
        //  I/O: Save Data  //
        //////////////////////
        void ThinShells::saveTree(const string &filename) const {
            string t_filename = filename;
            //if (filename.empty()) t_filename = concatFilePath((string)VIS_DIR, modelName, uniformDir, std::to_string(treeDepth), (string)"svo.obj");
            //if (filename.empty()) t_filename = concatFilePath((string)VIS_DIR, modelName, uniformDir, std::to_string(treeDepth));
            if (filename.empty())
                t_filename = concatFilePath((string) VIS_DIR, modelName, uniformDir, noiseDir,
                                            std::to_string(treeDepth));

            svo.saveSVO(t_filename);
        }

        void ThinShells::saveIntersections(const string &filename, const vector<Vector3d> &intersections) const {
            checkDir(filename);
            std::ofstream out(filename);
            if (!out) {
                fprintf(stderr, "[I/O] Error: File %s could not be opened!", filename.c_str());
                return;
            }

            for (const auto &p: intersections)
                out << p.x() << " " << p.y() << " " << p.z() << std::endl;
            out.close();
        }

        void ThinShells::saveIntersections(const string &filename_1, const string &filename_2) const {
            string t_filename = filename_1;
            if (filename_1.empty())
                t_filename = concatFilePath((string) VIS_DIR, modelName, uniformDir, noiseDir,
                                            std::to_string(treeDepth), (string) "edgeInter.xyz");
            std::cout << "-- Save mesh EDGES and octree Nodes to " << std::quoted(t_filename) << std::endl;
            saveIntersections(t_filename, edgeInterPoints);

            t_filename = filename_2;
            if (filename_2.empty())
                t_filename = concatFilePath((string) VIS_DIR, modelName, uniformDir, noiseDir,
                                            std::to_string(treeDepth), (string) "faceInter.xyz");
            std::cout << "-- Save mesh FACES and octree node EDGES to " << std::quoted(t_filename) << std::endl;
            saveIntersections(t_filename, faceInterPoints);
        }

        void ThinShells::saveSDFValue(const string &filename) const {
            string t_filename = filename;
            if (filename.empty())
                t_filename = concatFilePath((string) OUT_DIR, modelName, uniformDir, noiseDir,
                                            std::to_string(treeDepth), (string) "SDFValue.txt");

            checkDir(t_filename);
            std::ofstream out(t_filename);
            if (!out) {
                fprintf(stderr, "[I/O] Error: File %s could not open!\n", filename.c_str());
                return;
            }

            std::cout << "-- Save SDF value to " << std::quoted(t_filename) << std::endl;
            for (const auto &val: sdfVal)
                out << val << std::endl;
            out.close();
        }

        void ThinShells::saveCoefficients(const string &filename) const {
            string t_filename = filename;
            if (filename.empty())
                t_filename = concatFilePath((string) OUT_DIR, modelName, uniformDir, noiseDir,
                                            std::to_string(treeDepth), (string) "Coefficients.txt");

            checkDir(t_filename);
            std::ofstream out(t_filename);
            if (!out) {
                fprintf(stderr, "[I/O] Error: File %s could not open!\n", t_filename.c_str());
                return;
            }

            std::cout << "-- Save coefficients to " << std::quoted(t_filename) << std::endl;
            for (const auto &val: lambda)
                out << val << std::endl;
        }

        void ThinShells::saveLatentPoint(const string &filename) const {
            string t_filename = filename;
            if (filename.empty())
                t_filename = concatFilePath((string) OUT_DIR, modelName, uniformDir, noiseDir,
                                            std::to_string(treeDepth), (string) "latent_point.xyz");

            checkDir(t_filename);
            //std::ofstream out(t_filename);
            //std::ofstream out(t_filename, std::ios_base::app);
            std::ofstream out(t_filename, std::ofstream::out | std::ofstream::trunc);
            if (!out) {
                fprintf(stderr, "[I/O] Error: File %s could not open!\n", t_filename.c_str());
                return;
            }

            std::cout << "-- Save latent point to " << std::quoted(t_filename) << std::endl;

            gvis::writePointCloud_xyz(vertMat, out);
            out.close();
            out.open(t_filename, std::ofstream::app);
            gvis::writePointCloud_xyz(allInterPoints, out);
            out.close();
        }

        void ThinShells::saveBSplineValue(const string &filename) const {
            string t_filename = filename;
            if (filename.empty())
                t_filename = concatFilePath((string) OUT_DIR, modelName, uniformDir, noiseDir,
                                            std::to_string(treeDepth), (string) "BSplineValue.txt");

            checkDir(t_filename);
            std::ofstream out(t_filename);
            if (!out) {
                fprintf(stderr, "[I/O] Error: File %s could not open!\n", t_filename.c_str());
                return;
            }

            std::cout << "-- Save B-Spline value to " << std::quoted(t_filename) << std::endl;
            out << std::setiosflags(std::ios::fixed) << std::setprecision(9) << bSplineVal << std::endl;
            out.close();
        }

        //////////////////////
        //   Visualiztion   //
        //////////////////////
        void ThinShells::mcVisualization(const string &innerFilename, const Vector3i &innerResolution,
                                         const string &outerFilename, const Vector3i &outerResolution,
                                         const string &isoFilename, const Vector3i &isoResolution) {
            if (svo.numNodeVerts == 0) {
                printf("[MC] Warning: There is no valid Sparse Voxel Octree's node vertex, MarchingCubes is exited...\n");
                return;
            }
            if (depthMorton2Nodes.empty() || depthVert2Idx.empty()) prepareTestDS();

            Vector3d gridOrigin = modelBoundingBox.boxOrigin;
            Vector3d gridEnd = modelBoundingBox.boxEnd;
            Vector3d gridWidth = modelBoundingBox.boxWidth;

            Vector3i serializedRes(0, 0, 0);

            thrust::host_vector<double> gridSDF;
            auto determineGridSDF = [&](Vector3i resolution) {
                gridSDF.clear();
                gridSDF.shrink_to_fit();
                long long numVoxels = resolution.x() * resolution.y() * resolution.z();
                gridSDF.resize(numVoxels * 8, 0);

                if (svo.numNodeVerts == 0) {
                    printf("[MC] Warning: There is no valid Sparse Voxel Octree's node vertex, MarchingCubes is exited...\n");
                    return;
                }

                Vector3d voxelSize = Vector3d(gridWidth.x() / resolution.x(), gridWidth.y() / resolution.y(),
                                              gridWidth.z() / resolution.z());

#pragma omp parallel for
                for (int i = 0; i < numVoxels; ++i) {
                    uint x = i % resolution.x();
                    uint y = i % (resolution.x() * resolution.y()) / resolution.x();
                    uint z = i / (resolution.x() * resolution.y());

                    Vector3d voxelPos = gridOrigin + (Array3d(x * 1.0, y * 1.0, z * 1.0) * voxelSize.array()).matrix();
                    Vector3d corners[8];
                    corners[0] = voxelPos;
                    corners[1] = voxelPos + Vector3d(0, voxelSize.y(), 0);
                    corners[2] = voxelPos + Vector3d(voxelSize.x(), voxelSize.y(), 0);
                    corners[3] = voxelPos + Vector3d(voxelSize.x(), 0, 0);
                    corners[4] = voxelPos + Vector3d(0, 0, voxelSize.z());
                    corners[5] = voxelPos + Vector3d(0, voxelSize.y(), voxelSize.z());
                    corners[6] = voxelPos + Vector3d(voxelSize.x(), voxelSize.y(), voxelSize.z());
                    corners[7] = voxelPos + Vector3d(voxelSize.x(), 0, voxelSize.z());

                    for (int j = 0; j < 8; ++j) {
                        if ((corners[j].array() <= gridOrigin.array()).any() ||
                            (corners[j].array() >= gridEnd.array()).any())
                            gridSDF[i * 8 + j] = DINF;
                        else
                            gridSDF[i * 8 + j] = getPointBSplineVal(corners[j], true);
                    }
                }
            };

            //if (nodeWidthArray.empty())
            //{
            //	//nodeWidthArray.reserve(svo.numTreeNodes);
            //	auto& svoNodeArray = svo.svoNodeArray;
            //	std::transform(svoNodeArray.begin(), svoNodeArray.end(), std::back_inserter(nodeWidthArray),
            //		[](SVONode node) {
            //			return Eigen::Vector3d(node.width, node.width, node.width);
            //		});
            //}

            if (!outerFilename.empty() && outerShellIsoVal != -DINF && outerResolution.minCoeff() > 0) {
                std::cout << "\n[MC] Extract outer shell by MarchingCubes..." << std::endl;
#ifdef GPU_BSPLINEVAL
                MC::marching_cubes(svo.nodeVertexArray, svo.numTreeNodes, nodeWidthArray,
                                   svo.numNodeVerts, lambda, make_double3(gridOrigin), make_double3(gridWidth),
                                   make_uint3(outerResolution), outerShellIsoVal, outerFilename);
#else
                serializedRes = outerResolution;
                determineGridSDF(outerResolution);
//                std::cout << outerShellIsoVal << std::endl;
//                system("pause");
//                for (int i = 0; i < gridSDF.size(); ++i)
//                    if (gridSDF[i] < outerShellIsoVal)
//                        std::cout << gridSDF[i] << std::endl;
//                system("pause");
                MC::marching_cubes(make_uint3(outerResolution),
                                   make_double3(gridOrigin),
                                   make_double3(gridWidth),
                                   outerShellIsoVal,
                                   gridSDF, outerFilename);
#endif
                std::cout << "=====================\n";
            }

            if (!innerFilename.empty() && innerShellIsoVal != -DINF && innerResolution.minCoeff() > 0) {
                std::cout << "\n[MC] Extract inner shell by MarchingCubes..." << std::endl;
#ifdef GPU_BSPLINEVAL
                MC::marching_cubes(svo.nodeVertexArray, svo.numTreeNodes, nodeWidthArray,
                                   svo.numNodeVerts, lambda, make_double3(gridOrigin), make_double3(gridWidth),
                                   make_uint3(innerResolution), innerShellIsoVal, innerFilename);
#else
                if (innerResolution != serializedRes) {
                    determineGridSDF(innerResolution);
                    serializedRes = outerResolution;
                }
                MC::marching_cubes(make_uint3(innerResolution),
                                   make_double3(gridOrigin),
                                   make_double3(gridWidth),
                                   innerShellIsoVal,
                                   gridSDF, innerFilename);
#endif
                std::cout << "=====================\n";
            }

            if (!isoFilename.empty() && isoResolution.minCoeff() > 0) {
                std::cout << "\n[MC] Extract isosurface by MarchingCubes..." << std::endl;
#ifdef GPU_BSPLINEVAL
                MC::marching_cubes(svo.nodeVertexArray, svo.numTreeNodes, nodeWidthArray,
                                   svo.numNodeVerts, lambda, make_double3(gridOrigin), make_double3(gridWidth),
                                   make_uint3(isoResolution), .0, isoFilename);
#else
                if (isoResolution != serializedRes)
                    determineGridSDF(isoResolution);
                MC::marching_cubes(make_uint3(isoResolution),
                                   make_double3(gridOrigin),
                                   make_double3(gridWidth),
                                   .0,
                                   gridSDF, isoFilename);
#endif
                std::cout << "=====================\n";
            }
        }

        void ThinShells::textureVisualization(const string &filename) const {
            writeTexturedObjFile(filename, bSplineVal);
        }

        //////////////////////
        //    Application   //
        //////////////////////
        // ����ר��
        void ThinShells::prepareTestDS() {
            const vector<SVONode> &svoNodeArray = svo.svoNodeArray;
            const vector<node_vertex_type> &allNodeVertexArray = svo.nodeVertexArray;

            depthMorton2Nodes.clear();
            depthVert2Idx.clear();

            depthMorton2Nodes.resize(treeDepth);
            depthVert2Idx.resize(treeDepth);

            // ÿһ��ڵ�Ī��������(ȫ��)�±��ӳ��
            size_t _esumNodes = 0;
            const auto &depthNumNodes = svo.depthNumNodes;
            for (int d = 0; d < treeDepth; ++d) {
                vector<size_t> d_nodeIdx(depthNumNodes[d]);
                std::iota(d_nodeIdx.begin(), d_nodeIdx.end(), 0);

                std::transform(svoNodeArray.begin() + _esumNodes, svoNodeArray.begin() + _esumNodes + depthNumNodes[d],
                               d_nodeIdx.begin(), std::inserter(depthMorton2Nodes[d], depthMorton2Nodes[d].end()),
                               [_esumNodes](const SVONode &node, const size_t &idx) {
                                   return std::make_pair(node.mortonCode, _esumNodes + idx);
                               });
                _esumNodes += depthNumNodes[d];
            }

            // ÿ�㶥�㵽�����±�(ȫ��)��ӳ��
            const auto &esumDepthNodeVerts = svo.esumDepthNodeVerts;
            for (int d = 0; d < treeDepth; ++d) {
                const size_t d_numVerts = svo.depthNodeVertexArray[d].size();
                vector<size_t> d_numVertIdx(d_numVerts);
                std::iota(d_numVertIdx.begin(), d_numVertIdx.end(), 0);

                const size_t &d_esumNodeVerts = esumDepthNodeVerts[d]; // ����������exclusive scan
                std::transform(allNodeVertexArray.begin() + d_esumNodeVerts,
                               allNodeVertexArray.begin() + d_esumNodeVerts + d_numVerts,
                               d_numVertIdx.begin(), std::inserter(depthVert2Idx[d], depthVert2Idx[d].end()),
                               [d_esumNodeVerts](const node_vertex_type &val, const size_t &idx) {
                                   return std::make_pair(val.first, d_esumNodeVerts + idx);
                               });
            }
        }

        vector<int> ThinShells::multiPointQuery(const vector<Vector3d> &points, double &time, const int &session,
                                                const Test::type &choice) {
            test_type test = (test_type) choice;

            size_t numPoints = points.size();
            vector<int> result;
            if (innerShellIsoVal == -DINF || outerShellIsoVal == -DINF) {
                printf("Error: You must create shells first!\n");
                return result;
            }
            const vector<SVONode> &svoNodeArray = svo.svoNodeArray;
            VectorXd q_bSplineVal;

            const vector<node_vertex_type> &allNodeVertexArray = svo.nodeVertexArray;
            const size_t &numNodeVerts = svo.numNodeVerts;
            const Vector3d &boxOrigin = modelBoundingBox.boxOrigin;
            const Vector3d &boxEnd = modelBoundingBox.boxEnd;
            const Vector3d &boxWidth = modelBoundingBox.boxWidth;
            const Eigen::Array3d minRange = boxOrigin - boxWidth;
            const Eigen::Array3d maxRange = boxEnd + boxWidth;

            auto mt_cpuTest = [&]() {
#pragma omp parallel
                for (size_t i = 0; i < numPoints; ++i) {
                    const Vector3d &point = points[i];
                    if ((point.array() < minRange).any() || (point.array() > maxRange).any()) {
                        q_bSplineVal[i] = outerShellIsoVal;
                        continue;
                    }
                    double sum = .0;
#pragma omp parallel for reduction(+ : sum)
                    for (int j = 0; j < numNodeVerts; ++j) {
                        Vector3d nodeVert = allNodeVertexArray[j].first;
                        uint32_t nodeIdx = allNodeVertexArray[j].second;
                        double t = BaseFunction4Point(nodeVert, svoNodeArray[nodeIdx].width, point);
                        sum += lambda[j] * (BaseFunction4Point(nodeVert, svoNodeArray[nodeIdx].width, point));
                    }
                    q_bSplineVal[i] = sum;
                }
            };

            auto simd_cpuTest = [&]() {
#pragma omp parallel
                for (size_t i = 0; i < numPoints; ++i) {
                    const Vector3d &point = points[i];
                    if ((point.array() < minRange).any() || (point.array() > maxRange).any()) {
                        q_bSplineVal[i] = outerShellIsoVal;
                        continue;
                    }
                    double sum = .0;
#pragma omp simd simdlen(8)
                    for (int j = 0; j < numNodeVerts; ++j) {
                        Vector3d nodeVert = allNodeVertexArray[j].first;
                        uint32_t nodeIdx = allNodeVertexArray[j].second;
                        sum += lambda[j] * (BaseFunction4Point(nodeVert, svoNodeArray[nodeIdx].width, point));
                    }
                    q_bSplineVal[i] = sum;
                }
            };

            // ͨ���ҷ�Χ��b����ֵ
            auto mt_cpuTest_2 = [&]() {
#pragma omp parallel
                for (size_t i = 0; i < numPoints; ++i) {
                    const Vector3d &point = points[i];
                    if ((point.array() <= boxOrigin.array()).any() || (point.array() >= boxEnd.array()).any()) {
                        q_bSplineVal[i] = outerShellIsoVal;
                        continue;
                    }

                    double sum = 0.0;
                    Vector3i dis = getPointDis(point, boxOrigin, voxelWidth);
                    // �����и���(������Ե���Ӻʹ����)��Ӱ�췶Χ��
                    int maxOffset = 0;
                    int searchDepth = 0;
                    double searchNodeWidth = voxelWidth;
                    for (int i = 0; i < 3; ++i) {
                        // �ڱ�Ե����Ӱ�췶Χ��, ��Ϊ0����svo_gridSize[i] - 1��Ϊ�˺���Ī����ļ���
                        if (dis[i] <= -1) {
                            maxOffset = std::max(maxOffset, std::abs(dis[i]));
                            dis[i] = 0;
                        } else if (dis[i] >= svo_gridSize[i]) {
                            maxOffset = std::max(maxOffset, dis[i] - svo_gridSize[i] + 1);
                            dis[i] = svo_gridSize[i] - 1;
                        }
                    }
                    uint32_t pointMorton = morton::mortonEncode_LUT((uint16_t) dis.x(), (uint16_t) dis.y(),
                                                                    (uint16_t) dis.z());
                    maxOffset = nextPow2(maxOffset);
                    while (maxOffset >= 2) {
                        pointMorton /= 8;
                        ++searchDepth;
                        searchNodeWidth *= 2;
                        maxOffset >>= 1;
                    }

                    auto inDmPointsTraits = svo.mq_setInDomainPoints(pointMorton, modelOrigin, searchNodeWidth,
                                                                     searchDepth, depthMorton2Nodes, depthVert2Idx);
                    const int nInDmPointsTraits = inDmPointsTraits.size();

#pragma omp parallel for reduction(+ : sum) // forѭ���еı�����������з�������
                    for (int j = 0; j < nInDmPointsTraits; ++j) {
                        const auto &inDmPointTrait = inDmPointsTraits[j];
                        sum += lambda[std::get<2>(inDmPointTrait)] *
                               BaseFunction4Point(std::get<0>(inDmPointTrait), std::get<1>(inDmPointTrait), point);
                    }
                    q_bSplineVal[i] = sum;
                }
            };

            auto simd_cpuTest_2 = [&]() {
#pragma omp parallel
                for (size_t i = 0; i < numPoints; ++i) {
                    const Vector3d &point = points[i];
                    if ((point.array() < minRange).any() || (point.array() > maxRange).any()) {
                        q_bSplineVal[i] = outerShellIsoVal;
                        continue;
                    }

                    double sum = 0.0;
                    Vector3i dis = getPointDis(point, boxOrigin, voxelWidth);
                    // �����и���(������Ե���Ӻʹ����)��Ӱ�췶Χ��
                    int maxOffset = 0;
                    int searchDepth = 0;
                    double searchNodeWidth = voxelWidth;
                    for (int i = 0; i < 3; ++i) {
                        // �ڱ�Ե����Ӱ�췶Χ��, ��Ϊ0����svo_gridSize[i] - 1��Ϊ�˺���Ī����ļ���
                        if (dis[i] <= -1) {
                            maxOffset = std::max(maxOffset, std::abs(dis[i]));
                            dis[i] = 0;
                        } else if (dis[i] >= svo_gridSize[i]) {
                            maxOffset = std::max(maxOffset, dis[i] - svo_gridSize[i] + 1);
                            dis[i] = svo_gridSize[i] - 1;
                        }
                    }
                    uint32_t pointMorton = morton::mortonEncode_LUT((uint16_t) dis.x(), (uint16_t) dis.y(),
                                                                    (uint16_t) dis.z());
                    maxOffset = nextPow2(maxOffset);
                    while (maxOffset >= 2) {
                        pointMorton /= 8;
                        ++searchDepth;
                        searchNodeWidth *= 2;
                        maxOffset >>= 1;
                    }
                    auto inDmPoints = svo.mq_setInDomainPoints(pointMorton, modelOrigin, searchNodeWidth, searchDepth,
                                                               depthMorton2Nodes, depthVert2Idx);
                    const int nInDmPointsTraits = inDmPoints.size();

#pragma omp simd simdlen(8)
                    for (int j = 0; j < nInDmPointsTraits; ++j) {
                        const auto &inDmPointTrait = inDmPoints[j];
                        sum += lambda[std::get<2>(inDmPointTrait)] *
                               BaseFunction4Point(std::get<0>(inDmPointTrait), std::get<1>(inDmPointTrait), point);
                    }
                    q_bSplineVal[i] = sum;
                }
            };

            q_bSplineVal.resize(numPoints);
            TimerInterface *timer;
            createTimer(&timer);
            switch (test) {
                case Test::CPU:
                    printf("-- [Ours]: Using CPU\n");
                    if (depthMorton2Nodes.empty() || depthVert2Idx.empty()) prepareTestDS();
                    //mt_cpuTest_2();
                    for (int k = 0; k < session; ++k) {
                        printf("-- [Ours] [Session: %d/%d]", k + 1, session);
                        if (k != session - 1) printf("\r");
                        else printf("\n");

                        startTimer(&timer);

                        //mt_cpuTest();
                        mt_cpuTest_2();

                        stopTimer(&timer);
                    }
                    time = getAverageTimerValue(&timer) * 1e-3;
                    break;
                case Test::CPU_SIMD:
                    printf("-- [Ours]: Using CPU-SIMD\n");
                    if (depthMorton2Nodes.empty() || depthVert2Idx.empty()) prepareTestDS();
                    simd_cpuTest_2();
                    for (int k = 0; k < session; ++k) {
                        printf("-- [Ours] [Session: %d/%d]", k + 1, session);
                        if (k != session - 1) printf("\r");
                        else printf("\n");

                        startTimer(&timer);

                        //simd_cpuTest();
                        simd_cpuTest_2();

                        stopTimer(&timer);
                    }
                    time = getAverageTimerValue(&timer) * 1e-3;
                    break;
                default:
                case Test::CUDA:
                    printf("-- [Ours]: Using CUDA\n");
                    cuAcc::cpPointQuery(points.size(), svo.numNodeVerts, svo.numTreeNodes, minRange, maxRange,
                                        points, svo.nodeVertexArray, nodeWidthArray, lambda, outerShellIsoVal,
                                        q_bSplineVal);
                    for (int k = 0; k < session; ++k) {
                        printf("-- [Ours] [Session: %d/%d]", k + 1, session);
                        if (k != session - 1) printf("\r");
                        else printf("\n");

                        startTimer(&timer);

                        /// TODO: ��û�ĳɵ㳬��bbox�Ͳ���b����ֵ�İ汾
                        cuAcc::cpPointQuery(points.size(), svo.numNodeVerts, svo.numTreeNodes, minRange, maxRange,
                                            points, svo.nodeVertexArray, nodeWidthArray, lambda, outerShellIsoVal,
                                            q_bSplineVal);

                        stopTimer(&timer);
                    }
                    time = getAverageTimerValue(&timer) * 1e-3;
                    break;
            }
            deleteTimer(&timer);

            string t_filename = concatFilePath((string) VIS_DIR, modelName, uniformDir, std::to_string(treeDepth),
                                               (string) "temp_queryValue.txt");
            std::ofstream temp(t_filename);
            temp << std::setiosflags(std::ios::fixed) << std::setprecision(9) << q_bSplineVal << std::endl;

            std::transform(q_bSplineVal.begin(), q_bSplineVal.end(), std::back_inserter(result),
                           [=](const double &val) {
                               if (val >= outerShellIsoVal) return 1;
                               else if (val <= innerShellIsoVal) return -1;
                               else return 0;
                           });

            return result;
        }

        MatrixXd ThinShells::getPointNormal(const MatrixXd &queryPointMat) {
            const int numPoints = queryPointMat.rows();
            MatrixXd pointNormal(numPoints, 3);
            pointNormal.setZero();
            VectorXd q_bSplineVal(numPoints);
            q_bSplineVal.setZero();

            const Vector3d &boxOrigin = modelBoundingBox.boxOrigin;
            const Vector3d &boxEnd = modelBoundingBox.boxEnd;
            const Vector3d &boxWidth = modelBoundingBox.boxWidth;
            /*const Eigen::Array3d minRange = boxOrigin - boxWidth;
            const Eigen::Array3d maxRange = boxEnd + boxWidth;*/

#pragma omp parallel
            for (size_t i = 0; i < numPoints; ++i) {
                const Vector3d &point = queryPointMat.row(i);
                if ((point.array() < boxOrigin.array()).any() || (point.array() > boxEnd.array()).any()) {
                    std::cout << "Point is outside boundingbox!\n";
                    //std::cout << "point = " << point.transpose() << std::endl;
                    continue; // Ҫ��֤��boundingbox����
                }
                //if ((point.array() <= minRange).any() || (point.array() >= maxRange).any()) continue;

                Vector3d gradient;
                gradient.setZero();
                Vector3i dis = getPointDis(point, boxOrigin, voxelWidth);
                // �����и���(������Ե���Ӻʹ����)��Ӱ�췶Χ��
                //int maxOffset = 0;
                int searchDepth = 0;
                double searchNodeWidth = voxelWidth;
                uint32_t pointMorton = morton::mortonEncode_LUT((uint16_t) dis.x(), (uint16_t) dis.y(),
                                                                (uint16_t) dis.z());

                auto inDmPointsTraits = svo.mq_setInDomainPoints(pointMorton, modelOrigin, searchNodeWidth, searchDepth,
                                                                 depthMorton2Nodes, depthVert2Idx);
                const int nInDmPointsTraits = inDmPointsTraits.size();

                //#pragma omp parallel for reduction(+ : gradient) // forѭ���еı�����������з������ͣ�reduction�Ӿ��еı��������Ǳ�����������
                for (int j = 0; j < nInDmPointsTraits; ++j) {
                    const auto &inDmPointTrait = inDmPointsTraits[j];
                    gradient += lambda[std::get<2>(inDmPointTrait)] *
                                de_BaseFunction4Point(std::get<0>(inDmPointTrait), std::get<1>(inDmPointTrait), point);
                }
                pointNormal.row(i) = (gradient.normalized());
            }

            return pointNormal;
        }

        VectorXd ThinShells::getPointBSplineVal(const MatrixXd &queryPointMat) {
            const int numPoints = queryPointMat.rows();
            VectorXd q_bSplineVal;
            q_bSplineVal.setZero();

            if (depthMorton2Nodes.empty() || depthVert2Idx.empty()) prepareTestDS();

            Vector3d boxOrigin = modelBoundingBox.boxOrigin;
            Vector3d boxEnd = modelBoundingBox.boxEnd;
            Vector3d boxWidth = modelBoundingBox.boxWidth;
            Array3d minRange = boxOrigin - boxWidth;
            Array3d maxRange = boxEnd + boxWidth;

#pragma omp parallel
            for (size_t i = 0; i < numPoints; ++i) {
                const Vector3d &point = queryPointMat.row(i);
                if ((point.array() <= boxOrigin.array()).any() || (point.array() >= boxEnd.array()).any()) {
                    // std::cout << "[BSplineVal] Point is outside boundingbox!\n";
                    continue; // Ҫ��֤��boundingbox����
                }

                double sum = 0.0;
                Vector3i dis = getPointDis(point, boxOrigin, voxelWidth);
                // �����и���(������Ե���Ӻʹ����)��Ӱ�췶Χ��
                int maxOffset = 0;
                int searchDepth = 0;
                double searchNodeWidth = voxelWidth;
                //for (int i = 0; i < 3; ++i)
                //{
                //	// �ڱ�Ե����Ӱ�췶Χ��, ��Ϊ0����svo_gridSize[i] - 1��Ϊ�˺���Ī����ļ���
                //	if (dis[i] <= -1) { maxOffset = std::max(maxOffset, std::abs(dis[i])); dis[i] = 0; }
                //	else if (dis[i] >= svo_gridSize[i]) { maxOffset = std::max(maxOffset, dis[i] - svo_gridSize[i] + 1); dis[i] = svo_gridSize[i] - 1; }
                //}
                uint32_t pointMorton = morton::mortonEncode_LUT((uint16_t) dis.x(), (uint16_t) dis.y(),
                                                                (uint16_t) dis.z());
                /*maxOffset = nextPow2(maxOffset);
                while (maxOffset >= 2)
                {
                    pointMorton /= 8;
                    ++searchDepth;
                    searchNodeWidth *= 2;
                    maxOffset >>= 1;
                }*/

                auto inDmPointsTraits = svo.mq_setInDomainPoints(pointMorton, modelOrigin, searchNodeWidth, searchDepth,
                                                                 depthMorton2Nodes, depthVert2Idx);
                const int nInDmPointsTraits = inDmPointsTraits.size();

#pragma omp parallel for reduction(+ : sum) // forѭ���еı�����������з�������
                for (int j = 0; j < nInDmPointsTraits; ++j) {
                    const auto &inDmPointTrait = inDmPointsTraits[j];
                    sum += lambda[std::get<2>(inDmPointTrait)] *
                           BaseFunction4Point(std::get<0>(inDmPointTrait), std::get<1>(inDmPointTrait), point);
                }
                q_bSplineVal[i] = sum;
            }

            return q_bSplineVal;
        }

        double ThinShells::getPointBSplineVal(const Vector3d &queryPoint, bool dummy) const {
            double sum = 0.0;
            Vector3i dis = getPointDis(queryPoint, modelBoundingBox.boxOrigin, voxelWidth);
            int maxOffset = 0;
            int searchDepth = 0;
            double searchNodeWidth = voxelWidth;

            uint32_t pointMorton = morton::mortonEncode_LUT((uint16_t) dis.x(), (uint16_t) dis.y(),
                                                            (uint16_t) dis.z());

            auto inDmPointsTraits = svo.mq_setInDomainPoints(pointMorton, modelOrigin, searchNodeWidth, searchDepth,
                                                             depthMorton2Nodes, depthVert2Idx);
            int nInDmPointsTraits = inDmPointsTraits.size();

//#pragma omp parallel for reduction(+ : sum)
            for (int j = 0; j < nInDmPointsTraits; ++j) {
                const auto &inDmPointTrait = inDmPointsTraits[j];
                sum += lambda[std::get<2>(inDmPointTrait)] *
                       BaseFunction4Point(std::get<0>(inDmPointTrait), std::get<1>(inDmPointTrait), queryPoint);
            }

            return sum;
        }

    }
NAMESPACE_END(ITS)