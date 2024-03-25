#include "KNNHelper.h"
#include "ParticleMesh.h"
#include <omp.h>
#include <LBFGS.h>
#include <fstream>
#include <igl/knn.h>

using namespace LBFGSpp;

//void ParticleMesh::launchApp(const int& maxIterations, const int& numSearch, const std::string& out_file)
//{
//	lbfgs_optimization(maxIterations, numSearch, out_file);
//}

void ParticleMesh::lbfgs_optimization(const int& maxIterations, const int& _numSearch, const std::string& out_file)
{
	printf("[LBFGS] Optimizing particle system...\n");

	int numSearch = _numSearch;
	if (numSearch >= numParticles) numSearch = numParticles - 1;

	// Solver's param
	LBFGSParam<double> param;
	param.epsilon = 1e-5;
	param.max_iterations = 1;
	param.max_linesearch = 5000;

	// Project particles to the surface and compute normal
	Eigen::MatrixXd proj_particleMat;

	// Create function object
	auto optimize_fun = [&](const Eigen::VectorXd& before_particle, Eigen::VectorXd& grad)
	{
		double systemEnergy = .0;

		Eigen::MatrixXd before_particleMat(numParticles, 3);
		for (size_t i = 0; i < numParticles; ++i)
			before_particleMat.row(i) = Eigen::Vector3d(before_particle(i * 3), before_particle(i * 3 + 1), before_particle(i * 3 + 2));

		// Nearest point search
		KDTree kdTree(3, before_particleMat, 20);
		std::vector<Eigen::MatrixXd> neighPointList;
		knn_helper::getNeighborPoint(kdTree, before_particleMat, numSearch, neighPointList);

		// compute normal
		Eigen::MatrixXd normal = getPointNormal(before_particleMat);

		for (int i = 0; i < numParticles; ++i)
		{
			const Eigen::Vector3d i_particle = before_particleMat.row(i);

			// Compute gradient of i'th particle
			Eigen::Vector3d i_force;
			i_force.setZero();
			for (int j = 0; j < numSearch; ++j)
			{
				const Eigen::Vector3d j_neiParticle = neighPointList[i].row(j);

				const double& ij_energy = std::exp(-((i_particle - j_neiParticle).squaredNorm()) / (4 * theta * theta));
				systemEnergy += ij_energy;
				i_force += ((i_particle - j_neiParticle) / (2 * theta * theta)) * ij_energy;
			}

			// Project 'i_force' to the surface tangent
			const Eigen::Vector3d i_normal = normal.row(i);
			i_force = i_force - (i_force.dot(i_normal)) * i_normal;
			//i_force.normalize();

			// Update gradient
			grad(i * 3) = i_force.x();
			grad(i * 3 + 1) = i_force.y();
			grad(i * 3 + 2) = i_force.z();
		}
		std::cout << "systemEnergy = " << systemEnergy << std::endl;
		std::cout << "=========\n";
		return systemEnergy;
	};

	// Create solver
	LBFGSSolver<double, LineSearchBracketing> solver(param);

	double energy;

	Eigen::VectorXd particle_x;
	for (int iter = 1; iter <= maxIterations; ++iter)
	{
		if (iter == 1) { proj_particleMat = particleArray; /*normal = m_VN;*/ }

		Eigen::MatrixXd trans_proj_particleMat = proj_particleMat;
		trans_proj_particleMat.transposeInPlace();
		particle_x = (Eigen::Map<Eigen::VectorXd>(trans_proj_particleMat.data(), numParticles * 3));

		solver.minimize(optimize_fun, particle_x, energy);
		printf("-- [Iter: %d/%d] System energy = %lf\n", iter, maxIterations, energy);

		Eigen::MatrixXd particleMat(numParticles, 3);
#pragma omp parallel for
		for (int i = 0; i < numParticles; ++i)
			particleMat.row(i) = Eigen::Vector3d(particle_x(i * 3), particle_x(i * 3 + 1), particle_x(i * 3 + 2));
		proj_particleMat = getClosestPoint(particleMat);
	}

	if (!out_file.empty())
	{
		printf("[I/O] Output final partile to file: %s\n", out_file.c_str());
		std::ofstream out(out_file);
		for (size_t i = 0; i < numParticles; ++i)
			out << proj_particleMat.row(i) << std::endl;
		out.close();
	}
}
