#pragma once
#include "ModelDefine.h"
#include "BasicDataType.h"
#include "utils\Geometry.hpp"

class BaseModel
{
public:
	string uniformDir = "non-uniform";

protected:
	MXd m_V;
	MXi m_F;
	bool is2UnitCube;
	double scaleFactor;

	vector<V3d> modelVerts;
	vector<V3i> modelFaces;
	vector<Triangle<V3d>> modelTris;

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

	~BaseModel() {}

private:
	Eigen::Matrix4d calcTransformMatrix();
	Eigen::Matrix4d calcScaleMatrix();

public:
	void model2UnitCube();

	void unitCube2Model();

	void zoomModel();

public:
	vector<V2i> extractEdges();

	//void scaleMatrix(MXd V);

	void setBoundingBox(const double& scaleSize = 1);

	void setUniformBoundingBox();

	void setTriAttributes();

	Eigen::MatrixXd generateRandomPoints(const size_t& numPoints);

	Eigen::MatrixXd generateRandomPoints(const string& filename, const size_t& numPoints);

	// ��ȡ��ֵ��
	vector<vector<V3d>> extractIsoline(const vector<double>& scalarField, const double& val)const;

	// �з�ģ��
	std::pair<BaseModel, BaseModel> splitModelByIsoline(const vector<double>& scalarField, const double& val) const;

	vector<V3d> getVertices() const;

	vector<V3i> getFaces() const;

public:
	void readFile(const string& filename);

	void readOffFile(const string& filename);

	void readObjFile(const string& filename);

	void writeObjFile(const string& filename) const;

	void writeObjFile(const string& filename, const vector<V3d>& V, const vector<V3i>& F) const;

	void writeTexturedObjFile(const string& filename, const vector<PDD>& uvs) const;

	void writeTexturedObjFile(const string& filename, const VXd& uvs) const;

	void saveVertices(const string& filename, const vector<V3d>& verts);

	// �����ֵ��
	void saveIsoline(const string& filename, const vector<vector<V3d>>& isoline) const;
};