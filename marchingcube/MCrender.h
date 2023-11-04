#ifndef MCRENDER_H_
#define MCRENDER_H_

#include <windows.h>
#include <gl/GL.h>
#include <gl/glut.h>
//#include <QKeyEvent>
#include "MarchingCube.h"
#include "ImplicitFunction.h"
//#include "mycommon.h"


class MCImplicitRender
{
public:
	MCImplicitRender();
	MCImplicitRender(float minV1, float minV2, int X, TriMesh3D *mesh_, string file_);
	MCImplicitRender(float minV1, float minV2, int Nodes[3], TriMesh3D *mesh_, string file_);
	~MCImplicitRender();

	void InitData();
	void InitSurface();
	void RunMarchingCubesTestPotential(float& minValuePotential, std::vector<float> bg_node[3], std::vector<float>& mcPoints_inner_val);
	void RunMarchingCubesTestBoundary();
	void RunMarchingCubesTest();
	//void  OnKey(QKeyEvent* event);
	void DoMC();
	void TestMC();
	float  PotentialBoundary(Point3 p);
	float  Potential(Point3 p);

	void Remesh();
	void OuterTransferToOpenMesh();
	void InnerTransferToOpenMeshTest();
	void InnerTransferToOpenMesh(std::vector<float>& surface_node_x, std::vector<float>& surface_node_y, std::vector<float>& surface_node_z);
	void TransferToOpenMesh();
	void save_to_surface_node(std::vector<float>& surface_node_x, std::vector<float>& surface_node_y, std::vector<float>& surface_node_z);

	//render
	void DrawBoundar_(void) const;
	void DrawPoints(void) const;
	void DrawEdges(void) const;
	void DrawFaces(void) const;
	void DrawFlatWire(void);
	void renderWire(void);
	void renderFlat(void);
	void DrawSmoothShading(void);
	void DrawTransparent(void);
	void DrawWireFrame(void);
	void DrawFlatShading(void);
	void setMatirial(const float mat_diffuse[4], float mat_shininess);
	bool IsEmpty(void) const { return numOfTriangles; }
	void DrawBoundingBox(void);

	void SavePoints(const std::string& output_filename);

public:
	MyMesh  implicit_mesh;
	MyMesh  outer_mesh;
	MyMesh  inner_mesh;
	float minValuePotential;
	float minValueBoundary;
	int nX = 201, nY = 201, nZ = 201;
	Point3 voxelMin, voxelMax;
	TriMesh3D		*surfaceMesh;
	std::vector<mp4vector> mcPoints;
	std::vector<mp4vector> mcBoundary;
	TRIANGLE * Triangles;
	TRIANGLE * TrianglesBoundary;
	int numOfTriangles;
	int numOfTrianglesBoundary;

	float			r = 0.3, g = 0.3, b = 0.3, alpha;
	Point3		center;
	double		radius;
	double    meshSize;
	double t = 0.5;
	double c = 0.5;

	TVsplineFunction	*tvSpline_fun;
	BlobsFunction			blobs_fun;
};
#endif
