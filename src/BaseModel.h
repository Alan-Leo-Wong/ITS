#pragma once
#include "ModelDefine.h"
#include "BasicDataType.h"
#include "utils\Geometry.hpp"

class BaseModel
{
protected:
	MXd m_V;
	MXi m_F;

	vector<V3d> modelVerts;
	vector<V3i> modelFaces;
	vector<Triangle<V3d>> modelTris;

protected:
	string modelName;
	AABox<Eigen::Vector3d> modelBoundingBox;

	uint nModelVerts = 0;
	uint nModelTris = 0;

public:
	BaseModel() {};

	BaseModel(vector<V3d>verts, vector<V3i>faces) :modelVerts(verts), modelFaces(faces) {};

	BaseModel(const std::string& filename) 
	{ 
		readFile(filename);
		setUniformBoundingBox();
	}

	~BaseModel() {}

public:
	vector<V2i> extractEdges();

	void setBoundingBox(const double& scaleSize = 1);

	void setUniformBoundingBox();

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