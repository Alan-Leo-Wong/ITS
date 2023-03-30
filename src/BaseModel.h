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
	//! constructor and destructor
	BaseModel() {};

	BaseModel(const std::string& filename) { readFile(filename); }

	BaseModel(vector<V3d>verts, vector<V3i>faces) :modelVerts(verts), modelFaces(faces) {};

	~BaseModel() {}

public:
	// ��ȡģ�Ͷ���ӿ�
	vector<V3d> getVertices() const;

	vector<V3i> getFaces() const;

	// edges of the mesh
	vector<V2i> extractEdges();

	void setBoundingBox(const double& scaleSize);

	// ��ȡ��ֵ��
	vector<vector<V3d>> extractIsoline(const vector<double>& scalarField, const double& val)const;

	// �з�ģ��
	std::pair<BaseModel, BaseModel> splitModelByIsoline(const vector<double>& scalarField, const double& val) const;

public:
	//! IO
	void readFile(const string& filename);

	// ��ȡoff�ļ�
	void readOffFile(const string& filename);
	
	// ��ȡobj�ļ�
	void readObjFile(const string& filename);

	// дobj�ļ�
	void writeObjFile(const string& filename) const;

	void writeObjFile(const string& filename, const vector<V3d>& V, const vector<V3i>& F) const;

	// ��������
	void writeTexturedObjFile(const string& filename, const vector<PDD>& uvs) const;

	void writeTexturedObjFile(const string& filename, const VXd& uvs) const;

	void saveVertices(const string& filename, const vector<V3d>& verts);

	// �����ֵ��
	void saveIsoline(const string& filename, const vector<vector<V3d>>& isoline) const;
};