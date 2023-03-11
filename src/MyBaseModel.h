#pragma once
#include <algorithm>
#include <limits>
#include <Eigen\dense>
#include <sstream>   // �ַ���
#include <iostream>
#include <iomanip>   // set the precision of output data
#include <igl/read_triangle_mesh.h>
#include <igl/writeOBJ.h>
#include "BasicDataType.h"
#include "SharedPath.h"

class MyBaseModel
{
protected:
	vector<V3d> modelVerts;
	vector<V3i> modelFaces;
	MXd m_V;   // matrix form
	MXi m_F;

public:
	MyBaseModel() {};
	MyBaseModel(vector<V3d>verts, vector<V3i>faces) :modelVerts(verts), modelFaces(faces) {};
	// ��ȡģ�Ͷ���ӿ�
	vector<V3d> GetVertices()const;
	void SaveVertices(const string& filename, const vector<V3d>& verts);
	void ReadFile(const string& filename);
	// ��ȡobj�ļ�
	void ReadObjFile(const string& filename);
	// ��ȡoff�ļ�
	void ReadOffFile(const string& filename);
	// дobj�ļ�
	void WriteObjFile(const string& filename) const;
	void WriteObjFile(const string& filename, const vector<V3d>& V, const vector<V3i>& F) const;
	// edges of the mesh
	vector<V2i> extractEdges();
	// ��ȡ��ֵ��
	vector<vector<V3d>> ExtractIsoline(const vector<double>& scalarField, const double& val)const;
	// �����ֵ��
	void SaveIsoline(const string& filename, const vector<vector<V3d>>& isoline)const;
	// �з�ģ��
	std::pair< MyBaseModel, MyBaseModel>SplitModelByIsoline(const vector<double>& scalarField, const double& val)const;
	// ��������
	void WriteTexturedObjFile(const string& filename, const vector<PDD>& uvs)const;
	void WriteTexturedObjFile(const string& filename, const vector<double>& uvs)const;
};