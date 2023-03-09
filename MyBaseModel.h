#pragma once
#include <algorithm>
#include <limits>
#include <Eigen\dense>
#include <sstream>   // 字符流
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
	// 获取模型顶点接口
	vector<V3d> GetVertices()const;
	void SaveVertices(const string& filename, const vector<V3d>& verts);
	void ReadFile(const string& filename);
	// 读取obj文件
	void ReadObjFile(const string& filename);
	// 读取off文件
	void ReadOffFile(const string& filename);
	// 写obj文件
	void WriteObjFile(const string& filename) const;
	void WriteObjFile(const string& filename, const vector<V3d>& V, const vector<V3i>& F) const;
	// edges of the mesh
	vector<V2i> extractEdges();
	// 提取等值线
	vector<vector<V3d>> ExtractIsoline(const vector<double>& scalarField, const double& val)const;
	// 保存等值线
	void SaveIsoline(const string& filename, const vector<vector<V3d>>& isoline)const;
	// 切分模型
	std::pair< MyBaseModel, MyBaseModel>SplitModelByIsoline(const vector<double>& scalarField, const double& val)const;
	// 保存纹理
	void WriteTexturedObjFile(const string& filename, const vector<PDD>& uvs)const;
	void WriteTexturedObjFile(const string& filename, const vector<double>& uvs)const;
};