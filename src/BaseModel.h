#pragma once
#include "SharedPath.h"
#include "BasicDataType.h"

class BaseModel
{
protected:
	MXd m_V;
	MXi m_F;
	vector<V3d> modelVerts;
	vector<V3i> modelFaces;

public:
	BaseModel() {};
	BaseModel(vector<V3d>verts, vector<V3i>faces) :modelVerts(verts), modelFaces(faces) {};

	// ��ȡģ�Ͷ���ӿ�
	vector<V3d> getVertices()const;

	void saveVertices(const string& filename, const vector<V3d>& verts);

	void readFile(const string& filename);

	// ��ȡobj�ļ�
	void readObjFile(const string& filename);

	// ��ȡoff�ļ�
	void readOffFile(const string& filename);

	// дobj�ļ�
	void writeObjFile(const string& filename) const;

	void writeObjFile(const string& filename, const vector<V3d>& V, const vector<V3i>& F) const;

	// edges of the mesh
	vector<V2i> extractEdges();

	// ��ȡ��ֵ��
	vector<vector<V3d>> extractIsoline(const vector<double>& scalarField, const double& val)const;

	// �����ֵ��
	void saveIsoline(const string& filename, const vector<vector<V3d>>& isoline)const;

	// �з�ģ��
	std::pair< BaseModel, BaseModel> splitModelByIsoline(const vector<double>& scalarField, const double& val)const;

	// ��������
	void writeTexturedObjFile(const string& filename, const vector<PDD>& uvs)const;

	void writeTexturedObjFile(const string& filename, const vector<double>& uvs)const;
};