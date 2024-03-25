#pragma once
#include "BasicDataType.h"

class ParticleMesh
{
protected:
	double c_theta = 0.3;
	double embedOmiga = .0;
	double theta;

	size_t numParticles;
	MXd particleArray;

public:
	ParticleMesh() = default;

	//virtual void launchApp(const int& maxIterations, const int& numSearch, const std::string& out_file = "");

protected:
	virtual void lbfgs_optimization(const int& maxIterations, const std::string& out_file) = 0;
};