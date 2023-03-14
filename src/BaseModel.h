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

	// 获取模型顶点接口
	vector<V3d> getVertices()const;

	void saveVertices(const string& filename, const vector<V3d>& verts);

	void readFile(const string& filename);

	// 读取obj文件
	void readObjFile(const string& filename);

	// 读取off文件
	void readOffFile(const string& filename);

	// 写obj文件
	void writeObjFile(const string& filename) const;

	void writeObjFile(const string& filename, const vector<V3d>& V, const vector<V3i>& F) const;

	// edges of the mesh
	vector<V2i> extractEdges();

	// 提取等值线
	vector<vector<V3d>> extractIsoline(const vector<double>& scalarField, const double& val)const;

	// 保存等值线
	void saveIsoline(const string& filename, const vector<vector<V3d>>& isoline)const;

	// 切分模型
	std::pair< BaseModel, BaseModel> splitModelByIsoline(const vector<double>& scalarField, const double& val)const;

	// 保存纹理
	void writeTexturedObjFile(const string& filename, const vector<PDD>& uvs)const;

	void writeTexturedObjFile(const string& filename, const vector<double>& uvs)const;
};