#pragma once
#include "BasicDataType.h"
#include <nanoflann.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>

typedef nanoflann::KDTreeEigenMatrixAdaptor<MXd> KDTree;

namespace knn_helper
{
	inline void getClosestPoint(const KDTree& kdTree, const std::vector<V3d>& queryPointList,
		const MXd& modelVerts, std::vector<V3d>& resPointList)
	{
		// do a knn search
		const size_t        num_results = queryPointList.size();
		std::vector<size_t> ret_indexes(num_results);
		std::vector<int>    out_dists_sqr(num_results);

		nanoflann::KNNResultSet<int> resultSet(num_results);

		for (size_t i = 0; i < num_results; ++i)
		{
			resultSet.init(&ret_indexes[i], &out_dists_sqr[i]);
			kdTree.index_->findNeighbors(resultSet, queryPointList[i].data());
			resPointList.emplace_back(modelVerts.row(ret_indexes[i]));
		}
	}

	inline void getNeighborPoint(const KDTree& kdTree, const V3d& queryPoint, const size_t& query_i,
		const MXd& V, const size_t& num_results, std::vector<V3d>& neighPointList)
	{
		// do a knn search
		std::vector<size_t> ret_indexes(num_results + 1);
		std::vector<int>    out_dists_sqr(num_results + 1);

		nanoflann::KNNResultSet<int> resultSet(num_results + 1);

		resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
		kdTree.index_->findNeighbors(resultSet, queryPoint.data());
		for (size_t i = 0; i < num_results + 1; ++i)
		{
			if (ret_indexes[i] == query_i) continue;
			neighPointList.emplace_back(V.row(ret_indexes[i]));
			if (neighPointList.size() == num_results) break;
		}
	}

	inline void getNeighborPoint(const KDTree& kdTree, const MXd& queryPointMat,
		const size_t& num_results, std::vector<MXd>& neighPointList)
	{
		const size_t numPoints = queryPointMat.rows();
		for (size_t i = 0; i < numPoints; ++i)
		{
			// do a knn search
			std::vector<size_t> ret_indexes(num_results + 1);
			std::vector<int>    out_dists_sqr(num_results + 1);

			nanoflann::KNNResultSet<int> resultSet(num_results + 1);
			resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);

			kdTree.index_->findNeighbors(resultSet, queryPointMat.row(i).data());

			MXd i_neighPoints(num_results, 3);
			int validNeighbos = 0;
			for (size_t k = 0; k < num_results + 1; ++k)
			{
				if (ret_indexes[k] == i) continue;
				i_neighPoints.row(validNeighbos) = queryPointMat.row(ret_indexes[k]);
				++validNeighbos;
				if (validNeighbos == num_results) break;
			}
			neighPointList.emplace_back(i_neighPoints);
		}
	}
}