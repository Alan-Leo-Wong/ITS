#pragma once
#include "ModelDefine.h"
#include "BasicDataType.h"
#include "utils/Geometry.hpp"
#include <igl/AABB.h>

class BaseModel
{
public:
	string uniformDir = "non-uniform";
	string noiseDir = "non-noise";

protected:
	MXd m_V;
	MXd m_VN;
	MXi m_F;
	MXd m_FN;

	bool is2UnitCube;
	double scaleFactor;

	vector<V3d> modelVerts;
	vector<V3i> modelFaces;
	vector<Triangle<V3d>> modelTris;

	igl::AABB<MXd, 3> aabbTree;

protected:
	string modelName;
	AABox<Eigen::Vector3d> modelBoundingBox;

	uint nModelVerts = 0;
	uint nModelTris = 0;

private:
	void setModelAttributeVector();

public:
	BaseModel() {}

	BaseModel(vector<V3d>verts, vector<V3i>faces) :modelVerts(verts), modelFaces(faces), is2UnitCube(false), scaleFactor(1.0)
	{
		setUniformBoundingBox();
		setTriAttributes();
	}

	BaseModel(const std::string& filename)
	{
		readFile(filename);
		setModelAttributeVector();
		setUniformBoundingBox();
		setTriAttributes();
	}

	BaseModel(const std::string& filename, const bool& _is2UnitCube, const double& _scaleFactor)
		: is2UnitCube(_is2UnitCube), scaleFactor(_scaleFactor)
	{
		readFile(filename);
		if (_is2UnitCube) { uniformDir = "uniform"; model2UnitCube(); }
		setModelAttributeVector();
		setUniformBoundingBox();
		setTriAttributes();
	}

	BaseModel(const std::string& filename, const bool& _is2UnitCube, const double& _scaleFactor,
		const bool& _isAddNoise, const double& noisePercentage)
		: is2UnitCube(_is2UnitCube), scaleFactor(_scaleFactor)
	{
		readFile(filename);
		if (_is2UnitCube) { uniformDir = "uniform"; model2UnitCube(); }
		if (_isAddNoise)
		{
			printf("Add %lf%% noise on model...\n", noisePercentage * 100.0);
			noiseDir = (string)"noise_" + std::to_string(noisePercentage * 100.0);
			addNoise(noisePercentage);
		}
		setModelAttributeVector();
		setUniformBoundingBox();
		setTriAttributes();
	}

	~BaseModel() {}

public:
	vector<V3d> getModelVerts() { return modelVerts; }

	vector<V3i> getModelFaces() { return modelFaces; }

	vector<Triangle<V3d>> getModelTris() { return modelTris; }

private:
	Eigen::Matrix4d calcUnitCubeTransformMatrix();

	Eigen::Matrix4d calcTransformMatrix(const float& _scaleFactor);

	Eigen::Matrix3d calcScaleMatrix();

	void addNoise(const double& noisePercentage, const double& min_val = -0.1, const double& max_val = 0.1);

public:
	void model2UnitCube();

	void unitCube2Model();

	void zoomModel();

	void transformModel(const float& _scaleFactor);

public:
	vector<V2i> extractEdges();

	//void scaleMatrix(MXd V);

	void setBoundingBox(const float& _scaleFactor = 1);

	void setUniformBoundingBox();

	void setTriAttributes();

	Eigen::MatrixXd generateGaussianRandomPoints(const size_t& numPoints, const float& _scaleFactor, const float& dis);
	Eigen::MatrixXd generateUniformRandomPoints(const size_t& numPoints, const float& _scaleFactor, const float& dis);

	Eigen::MatrixXd generateGaussianRandomPoints(const string& filename, const size_t& numPoints, const float& _scaleFactor, const float& dis);
	Eigen::MatrixXd generateUniformRandomPoints(const string& filename, const size_t& numPoints, const float& _scaleFactor, const float& dis);

	vector<V3d> generateUniformRandomPoints(const string& filename, const size_t& numPoints, const double& _scaleFactor, const V3d& dis);
	vector<V3d> generateGaussianRandomPoints(const string& filename, const size_t& numPoints, const double& _scaleFactor, const V3d& dis);

	// 提取等值线
	vector<vector<V3d>> extractIsoline(const vector<double>& scalarField, const double& val)const;

	// 切分模型
	std::pair<BaseModel, BaseModel> splitModelByIsoline(const vector<double>& scalarField, const double& val) const;

	MXd getVertices() const;

	MXi getFaces() const;

protected:
	MXd getClosestPoint(const MXd& queryPointMat) const;

	virtual MXd getSurfacePointNormal(const MXd& queryPointMat);

public:
	void readFile(const string& filename);

	void readOffFile(const string& filename);

	void readObjFile(const string& filename);

	void writeObjFile(const string& filename) const;

	void writeObjFile(const string& filename, const vector<V3d>& V, const vector<V3i>& F) const;

	void writeTexturedObjFile(const string& filename, const vector<PDD>& uvs) const;

	void writeTexturedObjFile(const string& filename, const VXd& uvs) const;

	void saveVertices(const string& filename, const vector<V3d>& verts);

	// 保存等值线
	void saveIsoline(const string& filename, const vector<vector<V3d>>& isoline) const;
};