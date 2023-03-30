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

protected:
	string modelName;
	BoundingBox modelBoundingBox;

	uint nModelVerts = 0;
	uint nModelFaces = 0;

public:
	BaseModel() {};

	BaseModel(const std::string& filename) 
	{ 
		readFile(filename);
		setBoundingBox();
	}

	BaseModel(vector<V3d>verts, vector<V3i>faces) :modelVerts(verts), modelFaces(faces) {};

	~BaseModel() {}

public:
	vector<V2i> extractEdges();

	void setBoundingBox(const double& scaleSize = 0.1);

	// 提取等值线
	vector<vector<V3d>> extractIsoline(const vector<double>& scalarField, const double& val)const;

	// 切分模型
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

	// 保存等值线
	void saveIsoline(const string& filename, const vector<vector<V3d>>& isoline) const;
};